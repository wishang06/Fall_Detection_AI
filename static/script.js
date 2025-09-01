let cameraActive = false;
let statsInterval;

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearFallBtn = document.getElementById('clearFallBtn');
const videoFeed = document.getElementById('videoFeed');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const statusMessage = document.getElementById('statusMessage');
const fallAlert = document.getElementById('fallAlert');
const fallTime = document.getElementById('fallTime');


// Button event listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
clearFallBtn.addEventListener('click', clearFallAlert);

function showStatus(message, type = 'success') {
    statusMessage.innerHTML = `<div class="status-message status-${type}">${message}</div>`;
    setTimeout(() => {
        statusMessage.innerHTML = '';
    }, 3000);
}

async function startCamera() {
    try {
        const response = await fetch('/start_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            cameraActive = true;
            videoFeed.src = '/video_feed';
            videoFeed.style.display = 'block';
            videoPlaceholder.style.display = 'none';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            showStatus('Camera started successfully!', 'success');
            
            // Start polling for stats
            statsInterval = setInterval(updateStats, 1000);
        } else {
            showStatus('Failed to start camera: ' + data.message, 'error');
        }
    } catch (error) {
        showStatus('Error starting camera: ' + error.message, 'error');
    }
}

async function stopCamera() {
    try {
        const response = await fetch('/stop_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            cameraActive = false;
            videoFeed.style.display = 'none';
            videoPlaceholder.style.display = 'flex';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            showStatus('Camera stopped successfully!', 'success');
            
            // Stop polling for stats
            if (statsInterval) {
                clearInterval(statsInterval);
            }
        } else {
            showStatus('Failed to stop camera: ' + data.message, 'error');
        }
    } catch (error) {
        showStatus('Error stopping camera: ' + error.message, 'error');
    }
}

async function clearFallAlert() {
    try {
        const response = await fetch('/clear_fall_alert', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            fallAlert.style.display = 'none';
            clearFallBtn.style.display = 'none';
            showStatus('Fall alert cleared', 'success');
        } else {
            showStatus('Failed to clear fall alert', 'error');
        }
    } catch (error) {
        showStatus('Error clearing fall alert: ' + error.message, 'error');
    }
}

async function updateStats() {
    if (!cameraActive) return;
    
    try {
        const response = await fetch('/get_stats');
        const data = await response.json();
        
        // Handle fall detection alerts
        if (data.fall_detected && fallAlert.style.display === 'none') {
            // Show fall alert
            fallAlert.style.display = 'block';
            clearFallBtn.style.display = 'inline-block';
            
            // Update fall time
            if (data.fall_alert_time) {
                const fallDateTime = new Date(data.fall_alert_time);
                fallTime.textContent = `Fall detected at: ${fallDateTime.toLocaleString()}`;
            }
            
            // Play alert sound using Web Audio API
            try {
                playAlertSound();
            } catch (audioError) {
                console.log('Could not play alert sound:', audioError);
            }
        }
        
        // Hide fall alert if no longer detected
        if (!data.fall_detected && fallAlert.style.display === 'block') {
            fallAlert.style.display = 'none';
            clearFallBtn.style.display = 'none';
        }
        
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

function playAlertSound() {
    // Create a simple beep sound using Web Audio API
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800; // 800 Hz tone
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 1);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 1);
}


// Initialize button states
startBtn.disabled = false;
stopBtn.disabled = true;
