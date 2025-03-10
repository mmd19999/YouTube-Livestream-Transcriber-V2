// YouTube Livestream Transcriber - Main JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const clientCount = document.getElementById('client-count');
    const youtubeUrlInput = document.getElementById('youtube-url');
    const connectButton = document.getElementById('connect-button');
    const transcriptionContainer = document.getElementById('transcription-container');
    const topicsContainer = document.getElementById('topics-container');

    // Additional DOM elements for livestream info
    const livestreamSection = document.getElementById('livestream-section');
    let livestreamInfoElement = document.createElement('div');
    livestreamInfoElement.className = 'livestream-info';
    livestreamInfoElement.style.display = 'none';
    livestreamSection.appendChild(livestreamInfoElement);

    // Connection monitoring variables
    const heartbeatTimeout = 10000; // 10 seconds (reduced from 20)
    let lastHeartbeat = null;
    let heartbeatTimer = null;
    let connectionMonitorTimer = null;
    let isLivestreamConnected = false;
    let currentYoutubeUrl = '';

    // Configure socket connection
    const socket = io({
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
        timeout: 10000,
        autoConnect: true
    });

    // Make socket instance globally available for debug console
    window.socket = socket;

    // Start connection monitor
    startConnectionMonitor();

    // Socket.IO event handlers
    socket.on('connect', () => {
        updateConnectionStatus(true);
        console.log('Connected to server');
        $('#connection-status').text('Connected to server').removeClass('text-danger').addClass('text-success');
        $('#connect-button').prop('disabled', false);

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.info('Connected to server');
        }

        // If we were previously connected to a livestream, reconnect
        if (isLivestreamConnected && currentYoutubeUrl) {
            socket.emit('connect_to_livestream', { url: currentYoutubeUrl });
            if (window.debugConsole) {
                window.debugConsole.info(`Reconnecting to YouTube URL: ${currentYoutubeUrl}`);
            }
        }
    });

    socket.on('disconnect', () => {
        updateConnectionStatus(false);
        console.log('Disconnected from server');
        $('#connection-status').text('Disconnected from server').removeClass('text-success').addClass('text-danger');
        $('#connect-button').prop('disabled', true);

        // Clear any monitoring timers
        if (connectionMonitorTimer) clearInterval(connectionMonitorTimer);
        if (heartbeatTimer) clearInterval(heartbeatTimer);

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.error('Disconnected from server');
        }
    });

    socket.on('client_count', (data) => {
        updateClientCount(data.count);

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.info(`Client count updated: ${data.count} clients connected`);
        }
    });

    // Livestream event handlers
    socket.on('livestream_connected', (data) => {
        console.log('Livestream connected:', data);
        displayLivestreamInfo(data);
        isLivestreamConnected = true;

        // Update transcription container
        transcriptionContainer.innerHTML = '<p>Transcription will begin shortly...</p>';

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.info(`Livestream connected: ${data.title}`, data);
        }

        // Update last heartbeat time
        lastHeartbeat = Date.now();
    });

    // Handle heartbeat messages
    socket.on('livestream_heartbeat', (data) => {
        // Update last heartbeat time
        lastHeartbeat = Date.now();
        isLivestreamConnected = true;

        console.debug('Received heartbeat:', data);

        // Update UI to show healthy connection
        $('#connection-status')
            .text('Connected to livestream')
            .removeClass('text-danger text-warning')
            .addClass('text-success');

        // Log to debug console if available - but only every 10th heartbeat to avoid spam
        if (window.debugConsole && data.count % 10 === 0) {
            window.debugConsole.debug(`Received heartbeat #${data.count}`);
        }
    });

    socket.on('livestream_error', (data) => {
        console.error('Livestream error:', data);

        // Show error message
        transcriptionContainer.innerHTML = `
            <div class="error-message">
                <h3>Error</h3>
                <p>${data.error}</p>
            </div>
        `;

        // Hide livestream info if it was previously shown
        livestreamInfoElement.style.display = 'none';
        isLivestreamConnected = false;

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.error(`Livestream error: ${data.error}`, data);
        }
    });

    // Transcription event handlers
    socket.on('transcription_result', (data) => {
        console.log('Transcription result:', data);

        // Add transcription to container
        addTranscriptionText(data);

        // Update last heartbeat time
        lastHeartbeat = Date.now();

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.info(`Transcription received: ${data.text.substring(0, 50)}...`, data);
        }
    });

    socket.on('livestream_disconnected', () => {
        console.log('Disconnected from livestream');
        isLivestreamConnected = false;
        currentYoutubeUrl = '';

        // Re-enable connect button
        connectButton.disabled = false;
        connectButton.textContent = 'Connect';

        // Add message to transcription container
        const disconnectMessage = document.createElement('div');
        disconnectMessage.className = 'transcription-message';
        disconnectMessage.innerHTML = '<p><em>Disconnected from livestream</em></p>';
        transcriptionContainer.appendChild(disconnectMessage);

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.info('Disconnected from livestream');
        }
    });

    // Connect button event handler
    connectButton.addEventListener('click', () => {
        const youtubeUrl = youtubeUrlInput.value.trim();

        if (!youtubeUrl) {
            alert('Please enter a YouTube livestream URL');
            return;
        }

        // Save the current URL
        currentYoutubeUrl = youtubeUrl;

        // Clear previous content
        clearContainers();

        // Show loading state
        transcriptionContainer.innerHTML = '<p class="placeholder">Connecting to livestream...</p>';

        // Disable connect button while connecting
        connectButton.disabled = true;
        connectButton.textContent = 'Connecting...';

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.info(`Connecting to YouTube URL: ${youtubeUrl}`);
        }

        // Emit event to server to connect to YouTube livestream
        socket.emit('connect_to_livestream', { url: youtubeUrl });
    });

    // Helper functions
    function updateConnectionStatus(isConnected) {
        if (isConnected) {
            statusIndicator.className = 'connected';
            statusText.textContent = 'Connected';
        } else {
            statusIndicator.className = 'disconnected';
            statusText.textContent = 'Disconnected';
        }
    }

    function updateClientCount(count) {
        const text = count === 1 ? '1 client connected' : `${count} clients connected`;
        clientCount.textContent = `(${text})`;
    }

    function clearContainers() {
        transcriptionContainer.innerHTML = '';
        topicsContainer.innerHTML = '';
        livestreamInfoElement.style.display = 'none';
    }

    function displayLivestreamInfo(data) {
        // Re-enable connect button
        connectButton.disabled = false;
        connectButton.textContent = 'Connect';

        // Create livestream info HTML
        livestreamInfoElement.innerHTML = `
            <div class="info-header">
                <h3>${data.title}</h3>
                <span class="channel">${data.channel}</span>
            </div>
            <div class="info-details">
                <span class="status live">🔴 LIVE</span>
                <span class="viewers">${data.viewers || 'Unknown'} viewers</span>
                <button id="disconnect-button" class="disconnect-button">Disconnect</button>
            </div>
        `;

        // Add disconnect button event listener
        const disconnectButton = document.getElementById('disconnect-button');
        if (disconnectButton) {
            disconnectButton.addEventListener('click', () => {
                socket.emit('disconnect_from_livestream');
                disconnectButton.disabled = true;
                disconnectButton.textContent = 'Disconnecting...';
                isLivestreamConnected = false;
                currentYoutubeUrl = '';
            });
        }

        // Show livestream info
        livestreamInfoElement.style.display = 'block';
    }

    function addTranscriptionText(data) {
        // Check if this is the first transcription
        if (transcriptionContainer.querySelector('.placeholder')) {
            transcriptionContainer.innerHTML = '';
        }

        // Format timestamp with exact time (no rounding)
        let timestampText = '';
        if (data.start_time !== undefined) {
            const exactStartTime = parseFloat(data.start_time);
            const minutes = Math.floor(exactStartTime / 60);
            const seconds = Math.floor(exactStartTime % 60);
            timestampText = `[${minutes}:${seconds.toString().padStart(2, '0')}] `;
        }

        // Create a unique ID based on the exact timestamp
        const segmentId = `segment-${data.start_time}-${data.end_time}`;

        // Check if this exact segment already exists (for updates)
        let existingElement = document.getElementById(segmentId);

        if (existingElement) {
            // Only update if the new text is longer than the existing text (likely more complete)
            const existingText = existingElement.querySelector('.text').textContent;
            if (data.text.length > existingText.length) {
                existingElement.innerHTML = `
                    <span class="timestamp">${timestampText}</span>
                    <span class="text">${data.text}</span>
                `;
            }
        } else {
            // Create a new transcription element
            const transcriptionElement = document.createElement('div');
            transcriptionElement.className = 'transcription-item';
            transcriptionElement.id = segmentId;

            // Add the transcription text with timestamp
            transcriptionElement.innerHTML = `
                <span class="timestamp">${timestampText}</span>
                <span class="text">${data.text}</span>
            `;

            // Find the proper position to insert this element based on start_time
            let inserted = false;
            const existingSegments = Array.from(transcriptionContainer.querySelectorAll('.transcription-item'));

            for (let i = 0; i < existingSegments.length; i++) {
                const segmentParts = existingSegments[i].id.split('-');
                if (segmentParts.length >= 3) {
                    const existingStartTime = parseFloat(segmentParts[1]);
                    if (parseFloat(data.start_time) < existingStartTime) {
                        transcriptionContainer.insertBefore(transcriptionElement, existingSegments[i]);
                        inserted = true;
                        break;
                    }
                }
            }

            // If we didn't insert it in the middle, add it to the end
            if (!inserted) {
                transcriptionContainer.appendChild(transcriptionElement);
            }
        }

        // Scroll to bottom
        transcriptionContainer.scrollTop = transcriptionContainer.scrollHeight;
    }

    // Handler for full transcript updates
    socket.on('full_transcript_update', (data) => {
        console.log('Full transcript update:', data);

        // Clear the transcription container
        transcriptionContainer.innerHTML = '';

        // Process each segment in the transcript
        if (data.transcript && Array.isArray(data.transcript)) {
            // The transcript now comes pre-grouped and sorted by 15-second boundaries
            // so we just need to add each segment in order
            data.transcript.forEach(segment => {
                addTranscriptionText(segment);
            });
        }

        // Log to debug console if available
        if (window.debugConsole) {
            window.debugConsole.info(`Full transcript update: ${data.total_segments} segments`, data);
        }
    });

    function startConnectionMonitor() {
        // Clear any existing timers
        if (connectionMonitorTimer) clearInterval(connectionMonitorTimer);

        // Set last heartbeat to now when starting
        lastHeartbeat = Date.now();

        // Check connection every 5 seconds
        connectionMonitorTimer = setInterval(() => {
            const timeSinceLastHeartbeat = Date.now() - lastHeartbeat;

            console.debug(`Connection monitor: ${timeSinceLastHeartbeat}ms since last heartbeat`);

            // If no heartbeat for more than the timeout, try to reconnect
            if (timeSinceLastHeartbeat > heartbeatTimeout) {
                console.warn(`No heartbeat received for ${timeSinceLastHeartbeat}ms, reconnecting...`);

                // Update UI to show connection issue
                if (isLivestreamConnected) {
                    addLog('warning', 'Connection issue detected - attempting to reconnect...');
                    $('#connection-status').text('Reconnecting...').addClass('text-warning');

                    // Only attempt reconnection if we have a URL and were previously connected
                    if (currentYoutubeUrl && isLivestreamConnected) {
                        isLivestreamConnected = false;

                        // Try to reconnect to the livestream
                        reconnectToLivestream();
                    }
                }
            }
        }, 5000);
    }

    function reconnectToLivestream() {
        if (!currentYoutubeUrl) return;

        console.log(`Attempting to reconnect to: ${currentYoutubeUrl}`);

        // Connect to the previous URL
        socket.emit('connect_to_livestream', { youtube_url: currentYoutubeUrl });
    }

    function connectToLivestream(youtubeUrl) {
        // Store the URL for potential reconnection
        currentYoutubeUrl = youtubeUrl;

        // Reset the transcript
        $('#transcript').empty();

        // Show loading indicator
        $('#loading-indicator').show();
        $('#connection-status').text('Connecting...').removeClass('text-success text-danger').addClass('text-warning');

        // Connect to the livestream
        socket.emit('connect_to_livestream', { youtube_url: youtubeUrl });

        // Start monitoring the connection
        startConnectionMonitor();
    }
}); 