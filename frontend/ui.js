/**
 * ui.js - UI Updates and Main Application Logic
 * 
 * Handles:
 * - Initializing the application
 * - Updating the display with predictions
 * - Managing probability bars
 * - Status indicator updates
 */

class DigitRecognizerApp {
    constructor() {
        // Initialize components
        this.canvas = new DrawingCanvas('drawingCanvas', 'previewCanvas');
        this.api = new DigitAPI();
        
        // DOM elements
        this.predictionDigit = document.getElementById('predictionDigit');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.probabilityBars = document.getElementById('probabilityBars');
        this.clearBtn = document.getElementById('clearBtn');
        this.predictBtn = document.getElementById('predictBtn');
        this.statusIndicator = document.getElementById('statusIndicator');
        
        // State
        this.isLoading = false;
        
        this.init();
    }
    
    async init() {
        // Create probability bars
        this.createProbabilityBars();
        
        // Bind button events
        this.clearBtn.addEventListener('click', () => this.handleClear());
        this.predictBtn.addEventListener('click', () => this.handlePredict());
        
        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !this.isLoading) {
                this.handlePredict();
            }
            if (e.key === 'Escape') {
                this.handleClear();
            }
        });
        
        // Check API health
        await this.checkConnection();
        
        // Periodic health check
        setInterval(() => this.checkConnection(), 30000);
    }
    
    createProbabilityBars() {
        this.probabilityBars.innerHTML = '';
        
        for (let i = 0; i < 10; i++) {
            const row = document.createElement('div');
            row.className = 'bar-row';
            row.innerHTML = `
                <span class="bar-label">${i}</span>
                <div class="bar-track">
                    <div class="bar-fill" id="bar-${i}" style="width: 0%"></div>
                </div>
                <span class="bar-value" id="value-${i}">0%</span>
            `;
            this.probabilityBars.appendChild(row);
        }
    }
    
    async checkConnection() {
        const health = await this.api.checkHealth();
        
        if (health.success) {
            this.updateStatus('connected', health.modelLoaded ? 'Ready' : 'Model not loaded');
            
            if (!health.modelLoaded) {
                this.updateStatus('error', 'Train model first');
            }
        } else {
            this.updateStatus('error', 'Disconnected');
        }
    }
    
    updateStatus(state, text) {
        this.statusIndicator.className = 'status-indicator ' + state;
        this.statusIndicator.querySelector('.status-text').textContent = text;
    }
    
    handleClear() {
        this.canvas.clear();
        this.resetPrediction();
    }
    
    resetPrediction() {
        this.predictionDigit.textContent = '?';
        this.predictionDigit.classList.remove('success');
        this.confidenceValue.textContent = '--%';
        
        // Reset bars
        for (let i = 0; i < 10; i++) {
            const bar = document.getElementById(`bar-${i}`);
            const value = document.getElementById(`value-${i}`);
            bar.style.width = '0%';
            bar.classList.remove('highlight');
            value.textContent = '0%';
        }
    }
    
    async handlePredict() {
        if (this.isLoading) return;
        
        if (this.canvas.isEmpty()) {
            this.showMessage('Please draw a digit first');
            return;
        }
        
        // Set loading state
        this.isLoading = true;
        this.predictBtn.classList.add('loading');
        this.predictBtn.disabled = true;
        
        try {
            // Get pixel data
            const pixels = this.canvas.getPixelData();
            
            // Send to API
            const result = await this.api.predict(pixels);
            
            if (result.success) {
                this.displayResult(result);
            } else {
                this.showMessage(result.error || 'Prediction failed');
            }
        } catch (error) {
            this.showMessage('Error: ' + error.message);
        } finally {
            // Clear loading state
            this.isLoading = false;
            this.predictBtn.classList.remove('loading');
            this.predictBtn.disabled = false;
        }
    }
    
    displayResult(result) {
        // Update main prediction
        this.predictionDigit.textContent = result.digit;
        this.predictionDigit.classList.add('success');
        
        // Update confidence
        const confidencePercent = (result.confidence * 100).toFixed(1);
        this.confidenceValue.textContent = `${confidencePercent}%`;
        
        // Update probability bars
        const maxProb = Math.max(...result.probabilities);
        
        for (let i = 0; i < 10; i++) {
            const prob = result.probabilities[i];
            const percent = (prob * 100).toFixed(1);
            
            const bar = document.getElementById(`bar-${i}`);
            const value = document.getElementById(`value-${i}`);
            
            bar.style.width = `${prob * 100}%`;
            value.textContent = `${percent}%`;
            
            // Highlight the predicted digit
            if (i === result.digit) {
                bar.classList.add('highlight');
            } else {
                bar.classList.remove('highlight');
            }
        }
        
        // Remove success animation after a delay
        setTimeout(() => {
            this.predictionDigit.classList.remove('success');
        }, 300);
    }
    
    showMessage(message) {
        // Simple alert for now - could be replaced with a toast notification
        console.log(message);
        
        // Update prediction display to show error
        this.predictionDigit.textContent = '!';
        this.confidenceValue.textContent = message;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DigitRecognizerApp();
});