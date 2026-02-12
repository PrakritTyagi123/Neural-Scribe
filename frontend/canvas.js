/**
 * canvas.js - Canvas Drawing Logic
 * 
 * Handles user drawing on the canvas, including:
 * - Mouse and touch input
 * - Line drawing with smooth curves
 * - Downscaling to 28x28 for model input
 */

class DrawingCanvas {
    constructor(canvasId, previewCanvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.previewCanvas = document.getElementById(previewCanvasId);
        this.previewCtx = this.previewCanvas.getContext('2d');
        
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.hasDrawn = false;
        
        // Drawing settings - thicker line for better recognition
        this.lineWidth = 28;
        this.lineColor = '#000000';
        this.lineCap = 'round';
        this.lineJoin = 'round';
        
        this.init();
    }
    
    init() {
        // Set up canvas context
        this.ctx.lineCap = this.lineCap;
        this.ctx.lineJoin = this.lineJoin;
        this.ctx.lineWidth = this.lineWidth;
        this.ctx.strokeStyle = this.lineColor;
        
        // Clear canvas (white background)
        this.clear();
        
        // Bind event listeners
        this.bindEvents();
    }
    
    bindEvents() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.draw(e));
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseout', () => this.stopDrawing());
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', (e) => this.handleTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this.handleTouchMove(e));
        this.canvas.addEventListener('touchend', () => this.stopDrawing());
    }
    
    getCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }
    
    getTouchCoordinates(e) {
        const touch = e.touches[0];
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        
        return {
            x: (touch.clientX - rect.left) * scaleX,
            y: (touch.clientY - rect.top) * scaleY
        };
    }
    
    startDrawing(e) {
        this.isDrawing = true;
        this.hasDrawn = true;
        
        const coords = this.getCoordinates(e);
        this.lastX = coords.x;
        this.lastY = coords.y;
        
        // Draw a dot for single clicks
        this.ctx.beginPath();
        this.ctx.arc(coords.x, coords.y, this.lineWidth / 2, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Hide overlay
        this.hideOverlay();
        
        // Update preview
        this.updatePreview();
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const coords = this.getCoordinates(e);
        
        // Draw line
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(coords.x, coords.y);
        this.ctx.stroke();
        
        // Update last position
        this.lastX = coords.x;
        this.lastY = coords.y;
        
        // Update preview
        this.updatePreview();
    }
    
    stopDrawing() {
        this.isDrawing = false;
    }
    
    handleTouchStart(e) {
        e.preventDefault();
        this.isDrawing = true;
        this.hasDrawn = true;
        
        const coords = this.getTouchCoordinates(e);
        this.lastX = coords.x;
        this.lastY = coords.y;
        
        // Draw a dot
        this.ctx.beginPath();
        this.ctx.arc(coords.x, coords.y, this.lineWidth / 2, 0, Math.PI * 2);
        this.ctx.fill();
        
        this.hideOverlay();
        this.updatePreview();
    }
    
    handleTouchMove(e) {
        e.preventDefault();
        if (!this.isDrawing) return;
        
        const coords = this.getTouchCoordinates(e);
        
        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(coords.x, coords.y);
        this.ctx.stroke();
        
        this.lastX = coords.x;
        this.lastY = coords.y;
        
        this.updatePreview();
    }
    
    clear() {
        // Fill with white background
        this.ctx.fillStyle = '#ffffff';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Reset drawing color
        this.ctx.fillStyle = this.lineColor;
        this.ctx.strokeStyle = this.lineColor;
        
        this.hasDrawn = false;
        this.showOverlay();
        
        // Clear preview
        this.previewCtx.fillStyle = '#000000';
        this.previewCtx.fillRect(0, 0, 28, 28);
    }
    
    hideOverlay() {
        const overlay = document.getElementById('canvasOverlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }
    
    showOverlay() {
        const overlay = document.getElementById('canvasOverlay');
        if (overlay) {
            overlay.classList.remove('hidden');
        }
    }
    
    updatePreview() {
        // Get the 28x28 pixel data (already inverted in getPixelData)
        const pixelData = this.getPixelData();
        
        // Draw to preview canvas - show exactly what model sees
        const imageData = this.previewCtx.createImageData(28, 28);
        
        for (let i = 0; i < 784; i++) {
            // pixelData is already: 0=background, 255=digit (MNIST format)
            const value = pixelData[i];
            const idx = i * 4;
            imageData.data[idx] = value;     // R
            imageData.data[idx + 1] = value; // G
            imageData.data[idx + 2] = value; // B
            imageData.data[idx + 3] = 255;   // A
        }
        
        this.previewCtx.putImageData(imageData, 0, 0);
    }
    
    getPixelData() {
        // Create temporary canvas for downscaling
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Disable image smoothing for crisp downscaling
        tempCtx.imageSmoothingEnabled = true;
        tempCtx.imageSmoothingQuality = 'high';
        
        // Draw scaled down version
        tempCtx.drawImage(this.canvas, 0, 0, 28, 28);
        
        // Get pixel data
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const pixels = new Array(784);
        
        // Convert to grayscale (0-255)
        for (let i = 0; i < 784; i++) {
            const idx = i * 4;
            // Use standard grayscale formula
            const r = imageData.data[idx];
            const g = imageData.data[idx + 1];
            const b = imageData.data[idx + 2];
            
            // Grayscale conversion
            const gray = 0.299 * r + 0.587 * g + 0.114 * b;
            
            // Invert: white background (255) -> 0, black drawing (0) -> 255
            pixels[i] = 255 - gray;
        }
        
        return pixels;
    }
    
    isEmpty() {
        return !this.hasDrawn;
    }
}

// Export for use in other modules
window.DrawingCanvas = DrawingCanvas;