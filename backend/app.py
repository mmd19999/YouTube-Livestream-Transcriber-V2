#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main application for YouTube Livestream Transcriber

This module initializes the Flask application and Socket.IO server,
and handles client connections and requests.
"""

import os
import json
import logging
import time
import traceback
import threading
from datetime import datetime
from flask import Flask, render_template, send_from_directory, request, jsonify, session
from flask_socketio import SocketIO
from dotenv import load_dotenv
import debug_utils
from youtube_extractor import YouTubeExtractor
from audio_processor import AudioProcessor
from transcription import TranscriptionProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app")

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize Flask app
app = Flask(__name__, static_folder="../frontend/static")

# Configure session support with a secret key
app.secret_key = os.getenv("SECRET_KEY", "livestream-transcriber-secret-key")

# Configure Socket.IO with gevent WebSocket transport
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")

# Set socketio instance in debug_utils
debug_utils.set_socketio(socketio)

# Track connected clients
connected_clients = 0

# Global dictionary to store client data (key: client_id, value: dict with client data)
client_data = {}

# Initialize YouTube extractor
youtube_extractor = YouTubeExtractor()

# Initialize Audio processor
audio_processor = AudioProcessor()

# Initialize Transcription processor with memory
transcription_processor = TranscriptionProcessor(api_key=openai_api_key)

# Initialize TranscriptionMemory (already done inside TranscriptionProcessor)
debug_utils.emit_debug("Initialized enhanced transcription system with memory", "info")


@app.route("/")
def index():
    """Serve the main index.html page"""
    debug_utils.emit_debug(f"Serving index page to {request.remote_addr}", "info")
    return send_from_directory("../frontend", "index.html")


@socketio.on("connect")
def handle_connect():
    """Handle client connection event"""
    global connected_clients
    connected_clients += 1
    client_info = {
        "id": request.sid,
        "ip": request.remote_addr,
        "user_agent": request.headers.get("User-Agent", "Unknown"),
    }
    logger.info(f"Client connected. Total clients: {connected_clients}")
    socketio.emit("client_count", {"count": connected_clients})

    # Emit debug message
    debug_utils.emit_debug(
        f"Client connected from {request.remote_addr}",
        "info",
        {"client_count": connected_clients, "client_info": client_info},
    )


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnect event"""
    try:
        client_id = request.sid
        debug_utils.emit_debug(f"Client disconnecting: {client_id}", "info")

        if client_id in client_data:
            debug_utils.emit_debug(f"Cleaning up resources for {client_id}", "info")

            # Create a copy of the client data to work with while removing it
            client_info = client_data[client_id].copy()

            # Explicitly stop resources before removing data
            if "stop_event" in client_info:
                debug_utils.emit_debug(f"Setting stop event for {client_id}", "info")
                client_info["stop_event"].set()

                # Explicitly stop audio and transcription processing
                audio_processor.stop_extraction()
                transcription_processor.stop_processing()

                # Now remove client data after stopping processing
                if client_id in client_data:
                    del client_data[client_id]

                # To avoid issues with thread synchronization, use a background thread for cleanup
                def delayed_cleanup():
                    try:
                        debug_utils.emit_debug(
                            f"Running delayed cleanup for {client_id}", "info"
                        )
                        # Wait briefly to allow threads to terminate
                        time.sleep(2)

                        # Check if monitoring thread is still running
                        if (
                            "monitoring_thread" in client_info
                            and client_info["monitoring_thread"].is_alive()
                        ):
                            debug_utils.emit_debug(
                                f"Waiting for monitoring thread to terminate", "info"
                            )
                            # We don't need to join explicitly as daemon threads will be terminated when the process exits

                        # Additional cleanup
                        audio_processor._cleanup_temp_files()
                        transcription_processor._cleanup_temp_files()

                        debug_utils.emit_debug(
                            f"Delayed cleanup for {client_id} completed", "info"
                        )
                    except Exception as e:
                        debug_utils.emit_debug(
                            f"Error in delayed cleanup for {client_id}: {str(e)}",
                            "error",
                        )

                # Start a background thread for cleanup
                cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
                cleanup_thread.start()
            else:
                # If there's no stop event, we can just remove the client data
                if client_id in client_data:
                    del client_data[client_id]
    except Exception as e:
        debug_utils.emit_debug(f"Error handling disconnect: {str(e)}", "error")

    # Update connected client count
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    debug_utils.emit_debug(
        f"Client disconnected from {request.remote_addr}",
        "info",
        {"client_count": connected_clients},
    )


@socketio.on("connect_to_livestream")
def handle_connect_to_livestream(data):
    """Handle client request to connect to a YouTube livestream"""
    try:
        url = data.get("url")
        if not url:
            debug_utils.emit_debug("No URL provided", "error")
            socketio.emit("livestream_error", {"error": "No URL provided"})
            return

        debug_utils.emit_debug(f"Connecting to livestream: {url}", "info")

        # Create a stop event for controlled shutdown
        stop_event = threading.Event()

        # Create thread monitoring flag
        processing_active = threading.Event()
        processing_active.set()  # Start as active

        # Store client data in the global dictionary
        client_id = request.sid
        client_data[client_id] = {
            "stop_event": stop_event,
            "sid": client_id,
            "processing_active": processing_active,
            "thread_monitoring_started": False,
            "last_full_update": 0,  # Initialize last full update timestamp
        }

        debug_utils.emit_debug(f"Stored client data for {client_id}", "info")

        # Validate YouTube URL
        if not youtube_extractor.validate_youtube_url(url):
            debug_utils.emit_debug(f"Invalid YouTube URL: {url}", "error")
            socketio.emit("livestream_error", {"error": "Invalid YouTube URL"})
            return

        # Get stream info
        stream_info = youtube_extractor.get_stream_info(url)
        if not stream_info:
            debug_utils.emit_debug(f"Failed to get stream info for URL: {url}", "error")
            socketio.emit("livestream_error", {"error": "Failed to get stream info"})
            return

        # Get audio URL
        audio_url = youtube_extractor.get_audio_url(url)
        if not audio_url:
            debug_utils.emit_debug(
                f"Failed to get audio URL for stream: {url}", "error"
            )
            socketio.emit("livestream_error", {"error": "Failed to get audio URL"})
            return

        # Start audio processing
        debug_utils.emit_debug(f"Starting audio processing for URL: {url}", "info")

        # Define callback function for transcription results
        def transcription_callback(result):
            try:
                # Reset the monitoring timer each time we get a result
                if client_id in client_data:
                    client_data[client_id]["processing_active"].set()

                if result and "text" in result:
                    # Get additional metadata
                    is_merged = result.get("is_merged", False)
                    is_secondary_pass = result.get("is_secondary_pass", False)
                    pass_num = result.get("pass", 1)

                    debug_utils.emit_debug(
                        f"Transcription result (pass {pass_num}): {len(result['text'])} characters",
                        "info",
                        {
                            "start_time": result.get("start_time"),
                            "end_time": result.get("end_time"),
                            "text_length": len(result["text"]),
                            "is_merged": is_merged,
                            "is_secondary_pass": is_secondary_pass,
                        },
                    )

                    # Emit the transcription result to the client with enhanced metadata
                    socketio.emit(
                        "transcription_result",
                        {
                            "text": result["text"],
                            "start_time": result.get("start_time"),
                            "end_time": result.get("end_time"),
                            "is_merged": is_merged,
                            "is_secondary_pass": is_secondary_pass,
                            "pass": pass_num,
                            "timestamp": time.time(),
                        },
                    )

                    # Also emit the full transcript periodically for context
                    # Always emit for merged segments, and periodic updates for non-merged segments
                    if is_merged or (
                        time.time()
                        - client_data.get(client_id, {}).get("last_full_update", 0)
                        > 15
                    ):
                        # Get the full transcript with metadata
                        full_transcript = (
                            transcription_processor.memory.get_transcript_with_metadata(
                                use_timestamps=True
                            )
                        )

                        # Emit the full transcript to the client
                        socketio.emit(
                            "full_transcript_update",
                            {
                                "transcript": full_transcript,
                                "total_segments": len(full_transcript),
                                "total_words": transcription_processor.memory.total_words,
                                "merged_segments": transcription_processor.memory.merged_segments,
                                "timestamp": time.time(),
                            },
                        )

                        # Update the last full update timestamp
                        if client_id in client_data:
                            client_data[client_id]["last_full_update"] = time.time()
            except Exception as e:
                debug_utils.emit_debug(
                    f"Error handling transcription result: {str(e)}",
                    "error",
                    {"error": str(e)},
                )

        # Start transcription processing
        debug_utils.emit_debug("Starting transcription processing", "info")
        transcription_processor.start_processing(transcription_callback)

        # Define callback function for processed audio chunks
        def audio_chunk_callback(audio_data, start_time, end_time):
            try:
                debug_utils.emit_debug(
                    f"Received audio chunk: {start_time:.2f}s to {end_time:.2f}s, size: {len(audio_data)} bytes",
                    "info",
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "size": len(audio_data),
                    },
                )

                # Enhance the audio quality
                enhanced_audio = audio_processor.enhance_audio(audio_data)
                debug_utils.emit_debug(
                    f"Enhanced audio chunk: {start_time:.2f}s to {end_time:.2f}s, size: {len(enhanced_audio)} bytes",
                    "info",
                )

                # Add the enhanced audio chunk to the transcription queue
                transcription_processor.add_audio_chunk(
                    enhanced_audio, start_time, end_time
                )

                # Also process with multiple formats (for second pass)
                audio_formats = audio_processor.create_multi_format_audio(audio_data)
                debug_utils.emit_debug(
                    f"Created {len(audio_formats)} formats for audio chunk: {start_time:.2f}s to {end_time:.2f}s",
                    "info",
                )

                # Add each format to the transcription queue for multi-pass processing
                for i, format_data in enumerate(audio_formats):
                    # Use a slightly different start/end time to differentiate from original chunk
                    format_start = start_time + 0.001 * (i + 1)
                    format_end = end_time + 0.001 * (i + 1)
                    transcription_processor.add_multi_format_audio_chunk(
                        format_data["data"],
                        format_start,
                        format_end,
                        format_data["sample_rate"],
                        format_data["channels"],
                    )

                # Send a heartbeat to keep the connection alive
                socketio.emit("livestream_heartbeat", {"timestamp": time.time()})

            except Exception as e:
                debug_utils.emit_debug(
                    f"Error processing audio chunk: {str(e)}",
                    "error",
                    {"error": str(e)},
                )

        # Start audio processing
        debug_utils.emit_debug(f"Starting audio processing for URL: {url}", "info")

        # Start audio processing thread
        thread = audio_processor.process_audio(
            url, audio_chunk_callback, stop_event=stop_event
        )

        # Define monitoring parameters
        monitor_interval = 10  # Check thread status every 10 seconds

        # Start a monitoring thread to detect stalled processing
        def monitor_processing_thread():
            debug_utils.emit_debug(
                f"Started thread monitoring for client {client_id}", "info"
            )
            while not stop_event.is_set():
                # Check if the audio extraction thread is still running
                if (
                    audio_processor.processing_thread is None
                    or not audio_processor.processing_thread.is_alive()
                ):
                    debug_utils.emit_debug(
                        "Audio extraction thread is not running, restarting...",
                        "warning",
                    )
                    # Restart audio extraction
                    audio_processor.process_audio(url, audio_chunk_callback, stop_event)

                # Check if the transcription thread is still running
                if (
                    transcription_processor.processing_thread is None
                    or not transcription_processor.processing_thread.is_alive()
                ):
                    debug_utils.emit_debug(
                        "Transcription thread is not running, restarting...", "warning"
                    )
                    # Restart transcription processing
                    transcription_processor.start_processing(transcription_callback)

                time.sleep(monitor_interval)
            debug_utils.emit_debug("Monitoring thread stopped", "info")

        # Start monitoring thread
        monitoring_thread = threading.Thread(
            target=monitor_processing_thread, daemon=True
        )
        monitoring_thread.start()

        # Store the thread reference in client_data
        client_data[client_id]["monitoring_thread"] = monitoring_thread

        # Set up a heartbeat to keep the connection alive
        def send_heartbeat():
            """Send periodic heartbeats to keep the connection alive"""
            heartbeat_interval = 2  # Send heartbeat every 2 seconds (was 5)
            debug_utils.emit_debug(
                f"Starting heartbeat thread with interval: {heartbeat_interval}s",
                "info",
            )

            # Get client ID from parent thread's context
            nonlocal client_id
            client_sid = client_id  # Capture the client sid for use in the thread

            # Get the stop event from the client_data dictionary
            while (
                client_sid in client_data
                and not client_data[client_sid]["stop_event"].is_set()
            ):
                try:
                    socketio.emit(
                        "livestream_heartbeat",
                        {"timestamp": time.time(), "status": "processing"},
                        room=client_sid,  # Use stored sid instead of request.sid
                    )
                    debug_utils.emit_debug("Sent livestream heartbeat", "debug")
                    time.sleep(heartbeat_interval)
                except Exception as e:
                    debug_utils.emit_debug(
                        f"Error in heartbeat thread: {str(e)}", "error"
                    )
                    time.sleep(1)  # Short delay before retrying on error

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=send_heartbeat, daemon=True)
        heartbeat_thread.start()

        # Emit success message to client
        socketio.emit(
            "livestream_connected",
            {
                "title": stream_info.get("title", "Unknown"),
                "channel": stream_info.get("channel", "Unknown"),
                "viewers": stream_info.get("viewers", "Unknown"),
            },
        )

    except Exception as e:
        debug_utils.emit_debug(
            f"Error connecting to livestream: {str(e)}", "error", {"error": str(e)}
        )
        socketio.emit("livestream_error", {"error": str(e)})


@socketio.on("disconnect_from_livestream")
def handle_disconnect_from_livestream():
    """Handle client request to disconnect from a YouTube livestream"""
    try:
        debug_utils.emit_debug("Disconnecting from livestream", "info")

        # Get client ID and set the stop event from client_data dictionary
        client_id = request.sid
        if client_id in client_data:
            debug_utils.emit_debug(f"Setting stop event for client {client_id}", "info")

            # Create local reference to stop_event to avoid race conditions
            stop_event = client_data[client_id].get("stop_event")
            if stop_event:
                stop_event.set()

            # First, stop adding new audio to the processing pipeline
            # This prevents new items being added to queue
            audio_processor.stop_extraction()
            debug_utils.emit_debug("Stopped audio extraction", "info")

            # Briefly wait for any in-flight audio chunks to be added to transcription queue
            time.sleep(0.5)

            # Then stop transcription processing, which processes existing queued items
            transcription_processor.stop_processing()
            debug_utils.emit_debug("Stopped transcription processing", "info")

            # Perform cleanup
            debug_utils.emit_debug(
                f"Cleaning up resources for client {client_id}", "info"
            )

            # Create a background thread for cleanup to avoid blocking
            def cleanup_resources():
                try:
                    # Allow time for processing threads to recognize stop event
                    time.sleep(1)

                    # Clean up temp files
                    audio_processor._cleanup_temp_files()
                    transcription_processor._cleanup_temp_files()

                    # Now it's safer to perform cleanup
                    if client_id in client_data:
                        debug_utils.emit_debug(
                            f"Removing client {client_id} data", "info"
                        )
                        del client_data[client_id]
                except Exception as e:
                    debug_utils.emit_debug(
                        f"Error in resource cleanup: {str(e)}", "error"
                    )

            # Start cleanup in background thread
            cleanup_thread = threading.Thread(target=cleanup_resources, daemon=True)
            cleanup_thread.start()

        # Notify the client that the disconnect was processed
        socketio.emit("livestream_disconnected", {"status": "disconnected"})
    except Exception as e:
        debug_utils.emit_debug(
            f"Error disconnecting from livestream: {str(e)}", "error"
        )
        socketio.emit("livestream_error", {"error": str(e)})


if __name__ == "__main__":
    app.logger.info("Starting YouTube Livestream Transcriber server on port 5052")
    app.logger.info(
        f"OpenAI API key loaded: {openai_api_key[:5]}...{openai_api_key[-5:]}"
    )
    debug_utils.emit_debug("Server started on port 5052", "info")
    socketio.run(app, host="0.0.0.0", port=5052, debug=True)
