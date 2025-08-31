from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import json
import threading
import time
from datetime import datetime, timedelta
from src.track import BodyData, BodyTracker

app = Flask(__name__)

# Initialize MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables
camera = None
pose_processor = None
counter = 0
stage = None
fall_detected = False
fall_alert_time = None
camera_settings = {
    'brightness': 50,
    'contrast': 50,
    'saturation': 50,
    'detection_confidence': 0.7,
    'tracking_confidence': 0.7
}

class FallDetector:
    def __init__(self):
        self.previous_positions = []
        self.velocity_history = []
        self.acceleration_history = []
        self.position_history = []
        self.max_history = 10
        self.fall_threshold_velocity = -2.0  # m/s downward
        self.fall_threshold_acceleration = -15.0  # m/s² downward
        self.ground_proximity_threshold = 0.3  # relative to frame height
        self.stability_threshold = 0.1  # for detecting if person is stable
        self.fall_detected = False
        self.fall_start_time = None
        self.recovery_time = 3.0  # seconds to wait before clearing fall alert
        
    def detect_fall(self, landmarks, frame_height, frame_width):
        global fall_detected, fall_alert_time
        
        if not landmarks:
            return False
            
        try:
            # Get key body points
            head = landmarks[mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Calculate center of mass (approximate)
            center_of_mass_y = (head.y + left_shoulder.y + right_shoulder.y + 
                               left_hip.y + right_hip.y) / 5.0
            center_of_mass_x = (head.x + left_shoulder.x + right_shoulder.x + 
                               left_hip.x + right_hip.x) / 5.0
            
            current_position = np.array([center_of_mass_x, center_of_mass_y])
            current_time = datetime.now()
            
            # Store position history
            self.position_history.append((current_position, current_time))
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
            
            # Calculate velocity and acceleration if we have enough history
            if len(self.position_history) >= 3:
                # Calculate velocity (change in position over time)
                prev_pos, prev_time = self.position_history[-2]
                dt = (current_time - prev_time).total_seconds()
                if dt > 0:
                    velocity = (current_position - prev_pos) / dt
                    self.velocity_history.append(velocity)
                    if len(self.velocity_history) > self.max_history:
                        self.velocity_history.pop(0)
                
                # Calculate acceleration if we have velocity history
                if len(self.velocity_history) >= 2:
                    prev_velocity = self.velocity_history[-2]
                    acceleration = (velocity - prev_velocity) / dt if dt > 0 else np.array([0, 0])
                    self.acceleration_history.append(acceleration)
                    if len(self.acceleration_history) > self.max_history:
                        self.acceleration_history.pop(0)
                    
                    # Fall detection logic
                    vertical_velocity = velocity[1]  # y-component (positive is down in image coordinates)
                    vertical_acceleration = acceleration[1]
                    
                    # Check for rapid downward movement
                    rapid_descent = vertical_velocity > abs(self.fall_threshold_velocity)
                    
                    # Check if person is close to ground (high y-coordinate)
                    near_ground = center_of_mass_y > (1.0 - self.ground_proximity_threshold)
                    
                    # Check body orientation (horizontal vs vertical)
                    head_to_hip_distance = abs(head.y - (left_hip.y + right_hip.y) / 2.0)
                    horizontal_orientation = head_to_hip_distance < 0.3  # Person is lying down
                    
                    # Detect sudden impact (high acceleration)
                    sudden_impact = vertical_acceleration > abs(self.fall_threshold_acceleration)
                    
                    # Fall detection conditions
                    if (rapid_descent and near_ground) or horizontal_orientation or sudden_impact:
                        if not self.fall_detected:
                            self.fall_detected = True
                            self.fall_start_time = current_time
                            fall_detected = True
                            fall_alert_time = current_time
                            return True
                    
                    # Clear fall detection after recovery time
                    if self.fall_detected and self.fall_start_time:
                        time_since_fall = (current_time - self.fall_start_time).total_seconds()
                        if time_since_fall > self.recovery_time:
                            # Check if person has recovered (standing upright)
                            if head_to_hip_distance > 0.4 and center_of_mass_y < 0.7:
                                self.fall_detected = False
                                fall_detected = False
                                fall_alert_time = None
                                
        except Exception as e:
            print(f"Fall detection error: {e}")
            
        return self.fall_detected

class CameraProcessor:
    def __init__(self):
        self.cap = None
        self.pose = None
        self.running = False
        self.body_tracker = BodyTracker(80.0)
        self.fall_detector = FallDetector()
        
    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        
        # Apply camera settings
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, camera_settings['brightness'] / 100.0)
        self.cap.set(cv2.CAP_PROP_CONTRAST, camera_settings['contrast'] / 100.0)
        self.cap.set(cv2.CAP_PROP_SATURATION, camera_settings['saturation'] / 100.0)
        
        self.pose = mp_pose.Pose(
            min_detection_confidence=camera_settings['detection_confidence'],
            min_tracking_confidence=camera_settings['tracking_confidence']
        )
        self.running = True
        return True
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
    
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def process_frame(self):
        global counter, stage
        
        if not self.running or not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip and recolor
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = self.pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #retrieve the list of world_landmark positions
        if results.pose_world_landmarks and results.pose_landmarks:
            world_landmarks = results.pose_world_landmarks.landmark
            self.body_tracker.update(world_landmarks, results.pose_landmarks.landmark)
            print(f"Stress: {self.body_tracker.force.get_mag()}N")
            print(f"Net upward Stress: {self.body_tracker.force.get_net()[2]}N")

        # Extract landmarks and process
        try:
            landmarks = results.pose_landmarks.landmark
            frame_height, frame_width = image.shape[:2]
            
            # Fall detection
            is_falling = self.fall_detector.detect_fall(landmarks, frame_height, frame_width)
            
            # Get coordinates (right leg)
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angle
            angle = self.calculate_angle(hip, knee, ankle)
            
            # Display angle
            cv2.putText(image, str(round(angle, 2)),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Squat detection logic with fixed thresholds
            if angle < 90:  # Fixed down threshold
                stage = "down"
            if angle > 160 and stage == "down":  # Fixed up threshold
                stage = "up"
                counter += 1
            
            # Show feedback - expand the info box for fall detection
            info_height = 130 if fall_detected else 90
            cv2.rectangle(image, (0, 0), (300, info_height), (0, 0, 0), -1)
            cv2.putText(image, f"Reps: {counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"Stage: {stage}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Fall detection alert
            if fall_detected:
                cv2.putText(image, "FALL DETECTED!", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # Add flashing red border for fall alert
                cv2.rectangle(image, (0, 0), (frame_width-1, frame_height-1), (0, 0, 255), 10)
            
        except Exception as e:
            print(f"Processing error: {e}")
            pass
        
        # Render pose landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        return image

# Initialize camera processor
camera = CameraProcessor()

def generate_frames():
    while True:
        frame = camera.process_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return app.send_static_file('index.html')
    #return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<path:filename>')
def send_script(filename):
    return app.send_static_file(filename)

@app.route('/start_camera', methods=['POST'])
def start_camera():
    if camera.start():
        return jsonify({'status': 'success', 'message': 'Camera started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start camera'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    camera.stop()
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    global counter, stage
    counter = 0
    stage = None
    return jsonify({'status': 'success', 'counter': counter})

@app.route('/get_stats')
def get_stats():
    return jsonify({
        'counter': counter,
        'stage': stage,
        'fall_detected': fall_detected,
        'fall_alert_time': fall_alert_time.isoformat() if fall_alert_time else None,
        'settings': camera_settings
    })

@app.route('/clear_fall_alert', methods=['POST'])
def clear_fall_alert():
    global fall_detected, fall_alert_time
    fall_detected = False
    fall_alert_time = None
    if camera and camera.fall_detector:
        camera.fall_detector.fall_detected = False
        camera.fall_detector.fall_start_time = None
    return jsonify({'status': 'success', 'message': 'Fall alert cleared'})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global camera_settings
    data = request.json
    
    # Update settings
    for key, value in data.items():
        if key in camera_settings:
            camera_settings[key] = value
    
    # Apply camera settings if camera is running
    if camera.running and camera.cap:
        camera.cap.set(cv2.CAP_PROP_BRIGHTNESS, camera_settings['brightness'] / 100.0)
        camera.cap.set(cv2.CAP_PROP_CONTRAST, camera_settings['contrast'] / 100.0)
        camera.cap.set(cv2.CAP_PROP_SATURATION, camera_settings['saturation'] / 100.0)
        
        # Restart pose processor with new confidence settings
        camera.pose.close()
        camera.pose = mp_pose.Pose(
            min_detection_confidence=camera_settings['detection_confidence'],
            min_tracking_confidence=camera_settings['tracking_confidence']
        )
    
    return jsonify({'status': 'success', 'settings': camera_settings})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
