from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import json
import threading
import time
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
camera_settings = {
    'brightness': 50,
    'contrast': 50,
    'saturation': 50,
    'detection_confidence': 0.7,
    'tracking_confidence': 0.7
}

class CameraProcessor:
    def __init__(self):
        self.cap = None
        self.pose = None
        self.running = False
        self.body_tracker = BodyTracker(80.0)
        
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
        if results.pose_world_landmarks:
            world_landmarks = results.pose_world_landmarks.landmark
            self.body_tracker.update(world_landmarks, 0.01)
            height = self.body_tracker.get_height(world_landmarks)
            if (height < 0.5):
                print(f"Client falled: {height}")
            else:
                print(f"Client not falled: {height}")

        # Extract landmarks and process
        try:
            landmarks = results.pose_landmarks.landmark

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
            
            # Show feedback
            cv2.rectangle(image, (0, 0), (225, 90), (0, 0, 0), -1)
            cv2.putText(image, f"Reps: {counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(image, f"Stage: {stage}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
        except:
            
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
        'settings': camera_settings
    })

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
