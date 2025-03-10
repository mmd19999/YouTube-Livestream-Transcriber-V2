// YouTube Livestream Transcriber - Debug Console

/**
 * Debug Console for YouTube Livestream Transcriber
 * Provides logging, filtering, and Socket.IO event handling for debug messages
 */
class DebugConsole {
    constructor() {
        // DOM Elements
        this.debugSection = document.getElementById('debug-section');
        this.debugContainer = document.getElementById('debug-container');
        this.debugToggle = document.getElementById('debug-toggle');
        this.debugClear = document.getElementById('debug-clear');
        this.debugFilters = document.querySelectorAll('.debug-filter');

        // Initialize
        this.init();
        this.setupEventListeners();
        this.setupSocketListeners();

        // Log initial message
        this.log('Debug console initialized', 'info');
    }

    /**
     * Initialize the debug console
     */
    init() {
        // Clear placeholder
        this.debugContainer.innerHTML = '';
    }

    /**
     * Set up event listeners for debug console controls
     */
    setupEventListeners() {
        // Toggle debug console visibility
        this.debugToggle.addEventListener('click', () => {
            this.debugSection.classList.toggle('collapsed');
        });

        // Clear debug console
        this.debugClear.addEventListener('click', () => {
            this.debugContainer.innerHTML = '';
            this.log('Console cleared', 'info');
        });

        // Filter debug messages
        this.debugFilters.forEach(filter => {
            filter.addEventListener('change', () => {
                this.applyFilters();
            });
        });
    }

    /**
     * Set up Socket.IO event listeners for debug messages
     */
    setupSocketListeners() {
        // Get socket instance from main.js
        const socket = window.socket;

        if (!socket) {
            this.log('Socket.IO not available', 'error');
            return;
        }

        // Listen for debug messages from server
        socket.on('debug_message', (data) => {
            this.log(data.message, data.type, data.data);
        });

        // Listen for API request/response events
        socket.on('api_request', (data) => {
            this.log(`API Request: ${data.endpoint}`, 'api', data);
        });

        socket.on('api_response', (data) => {
            this.log(`API Response: ${data.endpoint}`, 'api', data);
        });

        // Listen for LLM events
        socket.on('llm_process', (data) => {
            this.log(`LLM Process: ${data.action}`, 'llm', data);
        });

        // Listen for verification events
        socket.on('verification_process', (data) => {
            this.log(`Verification: ${data.status}`, 'info', data);
        });
    }

    /**
     * Log a message to the debug console
     * @param {string} message - The message to log
     * @param {string} type - The type of message (info, warn, error, api, llm)
     * @param {object} data - Optional data to include with the message
     */
    log(message, type = 'info', data = null) {
        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = `debug-message debug-type-${type}`;
        messageEl.dataset.type = type;

        // Create timestamp
        const timestamp = new Date();
        const timestampEl = document.createElement('div');
        timestampEl.className = 'debug-timestamp';
        timestampEl.textContent = this.formatTimestamp(timestamp);

        // Create content
        const contentEl = document.createElement('div');
        contentEl.className = 'debug-content';
        contentEl.textContent = message;

        // Add JSON data if provided
        if (data) {
            const jsonEl = document.createElement('pre');
            jsonEl.className = 'debug-json';
            jsonEl.textContent = JSON.stringify(data, null, 2);
            contentEl.appendChild(jsonEl);
        }

        // Assemble message
        messageEl.appendChild(timestampEl);
        messageEl.appendChild(contentEl);

        // Add to container
        this.debugContainer.appendChild(messageEl);

        // Scroll to bottom
        this.debugContainer.scrollTop = this.debugContainer.scrollHeight;

        // Apply filters
        this.applyFilters();

        // Also log to console for easier debugging
        this.logToConsole(message, type, data);
    }

    /**
     * Format timestamp for display
     * @param {Date} date - The date to format
     * @returns {string} - Formatted timestamp (HH:MM:SS.mmm)
     */
    formatTimestamp(date) {
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        const seconds = date.getSeconds().toString().padStart(2, '0');
        const milliseconds = date.getMilliseconds().toString().padStart(3, '0');
        return `${hours}:${minutes}:${seconds}.${milliseconds}`;
    }

    /**
     * Apply filters to debug messages
     */
    applyFilters() {
        // Get active filters
        const activeFilters = Array.from(this.debugFilters)
            .filter(filter => filter.checked)
            .map(filter => filter.dataset.type);

        // Apply filters to messages
        const messages = this.debugContainer.querySelectorAll('.debug-message');
        messages.forEach(message => {
            if (activeFilters.includes(message.dataset.type)) {
                message.classList.remove('debug-hidden');
            } else {
                message.classList.add('debug-hidden');
            }
        });
    }

    /**
     * Log message to browser console for easier debugging
     * @param {string} message - The message to log
     * @param {string} type - The type of message
     * @param {object} data - Optional data to include
     */
    logToConsole(message, type, data) {
        const consoleMessage = `[${type.toUpperCase()}] ${message}`;

        switch (type) {
            case 'error':
                console.error(consoleMessage, data || '');
                break;
            case 'warn':
                console.warn(consoleMessage, data || '');
                break;
            default:
                console.log(consoleMessage, data || '');
                break;
        }
    }

    /**
     * Convenience method for logging info messages
     * @param {string} message - The message to log
     * @param {object} data - Optional data to include
     */
    info(message, data = null) {
        this.log(message, 'info', data);
    }

    /**
     * Convenience method for logging warning messages
     * @param {string} message - The message to log
     * @param {object} data - Optional data to include
     */
    warn(message, data = null) {
        this.log(message, 'warn', data);
    }

    /**
     * Convenience method for logging error messages
     * @param {string} message - The message to log
     * @param {object} data - Optional data to include
     */
    error(message, data = null) {
        this.log(message, 'error', data);
    }

    /**
     * Convenience method for logging API messages
     * @param {string} message - The message to log
     * @param {object} data - Optional data to include
     */
    api(message, data = null) {
        this.log(message, 'api', data);
    }

    /**
     * Convenience method for logging LLM messages
     * @param {string} message - The message to log
     * @param {object} data - Optional data to include
     */
    llm(message, data = null) {
        this.log(message, 'llm', data);
    }
}

// Initialize debug console when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create global debug console instance
    window.debugConsole = new DebugConsole();
}); 