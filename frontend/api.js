/**
 * api.js - API Communication Module
 * 
 * Handles all communication with the FastAPI backend:
 * - Sending pixel data for prediction
 * - Health checks
 * - Error handling
 */

class DigitAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.isConnected = false;
    }
    
    /**
     * Check if the backend server is healthy
     * @returns {Promise<Object>} Health status
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/health`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            
            const data = await response.json();
            this.isConnected = true;
            
            return {
                success: true,
                status: data.status,
                modelLoaded: data.model_loaded,
                version: data.version
            };
        } catch (error) {
            this.isConnected = false;
            return {
                success: false,
                error: error.message
            };
        }
    }
    
    /**
     * Send pixel data to the backend for prediction
     * @param {number[]} pixels - Array of 784 pixel values (0-255)
     * @returns {Promise<Object>} Prediction result
     */
    async predict(pixels) {
        try {
            // Validate input
            if (!Array.isArray(pixels) || pixels.length !== 784) {
                throw new Error('Invalid pixel data: expected 784 values');
            }
            
            const response = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ pixels })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error: ${response.status}`);
            }
            
            const data = await response.json();
            
            return {
                success: data.success,
                digit: data.digit,
                confidence: data.confidence,
                probabilities: data.probabilities,
                error: data.error
            };
        } catch (error) {
            return {
                success: false,
                digit: -1,
                confidence: 0,
                probabilities: new Array(10).fill(0),
                error: error.message
            };
        }
    }
    
    /**
     * Get top-k predictions
     * @param {number[]} pixels - Array of 784 pixel values
     * @param {number} k - Number of top predictions
     * @returns {Promise<Object>} Top-k predictions
     */
    async predictTopK(pixels, k = 3) {
        try {
            const response = await fetch(`${this.baseUrl}/predict/top-k?k=${k}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ pixels })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }
}

// Export for use in other modules
window.DigitAPI = DigitAPI;