let cameraActive = false;
let statsInterval;

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const videoFeed = document.getElementById('videoFeed');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const statusMessage = document.getElementById('statusMessage');

// Settings sliders
const sliders = {
    brightness: document.getElementById('brightness'),
    contrast: document.getElementById('contrast'),
    saturation: document.getElementById('saturation'),
    detectionConfidence: document.getElementById('detectionConfidence'),
    trackingConfidence: document.getElementById('trackingConfidence')
};

// Initialize slider value displays
Object.keys(sliders).forEach(key => {
    const slider = sliders[key];
    const valueDisplay = slider.parentElement.querySelector('.slider-value');
    
    slider.addEventListener('input', function() {
        valueDisplay.textContent = this.value;
        updateSettings();
    });
});

// Button event listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
resetBtn.addEventListener('click', resetCounter);

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

async function resetCounter() {
    try {
        const response = await fetch('/reset_counter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showStatus('Counter reset successfully!', 'success');
        } else {
            showStatus('Failed to reset counter', 'error');
        }
    } catch (error) {
        showStatus('Error resetting counter: ' + error.message, 'error');
    }
}

async function updateStats() {
    if (!cameraActive) return;
    
    try {
        const response = await fetch('/get_stats');
        const data = await response.json();
        
        // Stats are now displayed on the video feed itself
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

async function updateSettings() {
    const settings = {};
    
    Object.keys(sliders).forEach(key => {
        const value = parseFloat(sliders[key].value);
        settings[key.replace(/([A-Z])/g, '_$1').toLowerCase()] = value;
    });

    try {
        const response = await fetch('/update_settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings)
        });
        
        const data = await response.json();
        
        if (data.status !== 'success') {
            console.error('Failed to update settings');
        }
    } catch (error) {
        console.error('Error updating settings:', error);
    }
}

// Initialize button states
startBtn.disabled = false;
stopBtn.disabled = true;
