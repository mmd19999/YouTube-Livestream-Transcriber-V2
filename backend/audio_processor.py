#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio Processor for YouTube Livestream Transcriber

This module handles audio extraction and processing from YouTube streams,
optimizing audio for the Whisper API and processing it in batches.
"""

import os
import time
import logging
import tempfile
import threading
import subprocess
import traceback
import shutil
from pathlib import Path
from queue import Queue
import ffmpeg
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from youtube_extractor import YouTubeExtractor
import debug_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("audio_processor")

# Constants
SAMPLE_RATE = 16000  # 16kHz for Whisper API
VOLUME_BOOST = 1.5  # Volume boost factor for quiet audio
CHUNK_DURATION = 15  # seconds (exactly 15 seconds for consistent display)
OVERLAP_DURATION = 10.0  # seconds (increased from 3.0 to 10.0 for more frequent chunks)
# Ensure we advance exactly 5 seconds (CHUNK_DURATION - OVERLAP_DURATION) per segment
# This creates overlapping 15-second chunks that advance every 5 seconds
ADVANCE_DURATION = CHUNK_DURATION - OVERLAP_DURATION  # This should be exactly 5.0
# Multiple audio formats to try for best results
AUDIO_FORMATS = [
    {"sample_rate": 16000, "channels": 1},  # Standard Whisper format
    {"sample_rate": 22050, "channels": 1},  # Higher quality
    {"sample_rate": 44100, "channels": 1},  # CD quality
]


class AudioProcessor:
    """Class to handle audio extraction and processing from YouTube streams."""

    def __init__(self, temp_dir=None):
        """Initialize the audio processor with an optional temporary directory."""
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.extractor = YouTubeExtractor()
        self.chunk_size = CHUNK_DURATION  # 15 seconds as defined in the constant
        self.chunk_callback = None
        self.processing_thread = None
        self.stop_extraction_event = threading.Event()
        self.current_temp_files = []

        # Standard timeouts to ensure consistency
        self.network_timeout = 30  # Network operation timeout in seconds
        self.process_timeout = 20  # Process execution timeout in seconds
        self.extraction_timeout = 15  # Audio extraction timeout in seconds

    def register_chunk_callback(self, callback):
        """Register a callback function to receive audio chunks."""
        self.chunk_callback = callback
        debug_utils.emit_debug("Registered chunk callback", "info")

    def process_audio(self, url, on_chunk=None, stop_event=None, max_duration=300):
        """Process audio from a URL."""
        if on_chunk:
            self.chunk_callback = on_chunk

        # Use self.stop_extraction_event if no stop_event provided
        if stop_event is None:
            self.stop_extraction_event.clear()  # Reset the event
            stop_event = self.stop_extraction_event

        # Check if it's a YouTube URL
        if self.extractor.validate_youtube_url(url):
            debug_utils.emit_debug(
                f"Started live stream processing thread for URL: {url}"
            )
            thread = threading.Thread(
                target=self._process_live_stream_thread,
                args=(url, stop_event, max_duration),
            )
            thread.daemon = True
            thread.start()
            self.processing_thread = thread  # Store the thread for later access
            return thread
        else:
            # Assume it's a direct audio URL
            debug_utils.emit_debug(f"Direct audio URL: {url}")
            audio_data = self._download_audio(url)
            if audio_data:
                self._process_audio_data(audio_data)
            return None

    def _process_live_stream_thread(self, url, stop_event, max_duration):
        """Process a YouTube livestream in a separate thread using direct FFmpeg streaming."""
        debug_utils.emit_debug(
            "Starting FFmpeg direct streaming for YouTube audio", "info"
        )

        # Create a temporary directory for storing segments
        segments_dir = os.path.join(self.temp_dir, f"livestream_{int(time.time())}")
        os.makedirs(segments_dir, exist_ok=True)
        debug_utils.emit_debug(
            f"Created temporary directory for segments: {segments_dir}", "info"
        )

        # Track consecutive failures
        consecutive_failures = 0
        max_consecutive_failures = 3

        # Track stream URL refresh attempts
        url_refresh_attempts = 0
        max_url_refresh_attempts = 5
        last_refresh_time = 0
        min_refresh_interval = 60  # Minimum seconds between refreshes

        try:
            # Get the current script's directory to find cookies file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cookies_file = os.path.join(current_dir, "youtube_cookies.txt")

            if os.path.exists(cookies_file):
                debug_utils.emit_debug(
                    f"Using YouTube cookies file: {cookies_file}", "info"
                )
            else:
                debug_utils.emit_debug(
                    f"YouTube cookies file not found: {cookies_file}", "warning"
                )
                cookies_file = None

            # Function to get stream URL - refactored for reuse
            def get_stream_url():
                nonlocal url, cookies_file
                debug_utils.emit_debug("Getting HLS stream URL from YouTube", "info")

                # Use yt-dlp to get format information without downloading
                cmd = [
                    "yt-dlp",
                    "--format",
                    "91",  # Use format 91 which we know is available for livestreams
                    "--get-url",  # Just print the URL, don't download
                    "--no-playlist",
                    "--no-check-certificate",
                    "--geo-bypass",
                ]

                # Add cookies parameter if available
                if cookies_file:
                    cmd.extend(["--cookies", cookies_file])

                # Add the URL at the end
                cmd.append(url)

                debug_utils.emit_debug(
                    f"Executing yt-dlp command: {' '.join(cmd)}", "info"
                )

                try:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.network_timeout,  # Increased timeout
                    )

                    if result.returncode != 0 or not result.stdout.strip():
                        debug_utils.emit_debug(
                            f"Failed to get HLS URL from YouTube: {result.stderr}",
                            "error",
                        )
                        return None

                    # Get the direct audio stream URL
                    stream_url = result.stdout.strip()
                    debug_utils.emit_debug(
                        f"Got HLS audio stream URL: {stream_url[:50]}...", "info"
                    )
                    return stream_url

                except subprocess.TimeoutExpired:
                    debug_utils.emit_debug(
                        "yt-dlp process timed out when getting stream URL", "error"
                    )
                    return None
                except Exception as e:
                    debug_utils.emit_debug(
                        f"Error getting stream URL: {str(e)}",
                        "error",
                        {"exception": traceback.format_exc()},
                    )
                    return None

            # Get initial stream URL
            stream_url = get_stream_url()
            if not stream_url:
                debug_utils.emit_debug(
                    "Could not get initial stream URL, aborting", "error"
                )
                return

            # Set up segment parameters
            segment_duration = (
                CHUNK_DURATION  # Changed from 3 to CHUNK_DURATION (15 seconds)
            )
            start_time = 0
            segment_index = 0

            # Process until stop_event is set or max_duration is reached
            while (not stop_event or not stop_event.is_set()) and (
                not max_duration or start_time < max_duration
            ):
                try:
                    # Check if we need to refresh the stream URL (every 10 minutes or after multiple failures)
                    current_time = time.time()
                    if (current_time - last_refresh_time > 600) or (
                        consecutive_failures >= max_consecutive_failures
                    ):
                        if current_time - last_refresh_time >= min_refresh_interval:
                            debug_utils.emit_debug(
                                f"Time to refresh stream URL (failures: {consecutive_failures})",
                                "info",
                            )

                            if url_refresh_attempts < max_url_refresh_attempts:
                                new_stream_url = get_stream_url()
                                if new_stream_url:
                                    stream_url = new_stream_url
                                    consecutive_failures = 0  # Reset failure counter
                                    url_refresh_attempts += 1
                                    last_refresh_time = current_time
                                    debug_utils.emit_debug(
                                        "Successfully refreshed stream URL", "info"
                                    )
                                else:
                                    debug_utils.emit_debug(
                                        "Failed to refresh stream URL, continuing with existing URL",
                                        "warning",
                                    )
                            else:
                                debug_utils.emit_debug(
                                    f"Reached max URL refresh attempts ({max_url_refresh_attempts})",
                                    "warning",
                                )

                    # Define output file paths
                    segment_file = os.path.join(
                        segments_dir, f"segment_{segment_index}.m4a"
                    )
                    wav_file = os.path.join(
                        segments_dir, f"segment_{segment_index}.wav"
                    )

                    debug_utils.emit_debug(
                        f"Processing segment {segment_index}: {start_time}s - {start_time+segment_duration}s",
                        "info",
                    )

                    # FFmpeg command to extract a segment from the HLS stream
                    # Modified for sequential extraction from livestream
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-reconnect",
                        "1",  # Enable reconnection
                        "-reconnect_streamed",
                        "1",  # Reconnect if stream fails
                        "-reconnect_delay_max",
                        "5",  # Max reconnection delay
                        "-ss",
                        str(
                            start_time
                        ),  # Add seek parameter to start from correct position
                        "-i",
                        stream_url,  # Input HLS stream
                        "-t",
                        str(segment_duration),  # Duration to capture
                        "-c:a",
                        "copy",  # Copy audio codec (no re-encoding)
                        "-v",
                        "warning",  # Reduced verbosity
                        "-y",  # Overwrite output
                        segment_file,  # Output file
                    ]

                    # Run FFmpeg to extract segment
                    debug_utils.emit_debug(
                        f"Running FFmpeg to extract segment {segment_index} at time {start_time}s with URL length: {len(stream_url)}",
                        "info",
                        {
                            "segment_index": segment_index,
                            "start_time": start_time,
                            "consecutive_failures": consecutive_failures,
                            "url_refresh_attempts": url_refresh_attempts,
                        },
                    )

                    # Increased timeout for initial segment capture, but still reasonable
                    try:
                        ffmpeg_result = subprocess.run(
                            ffmpeg_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=self.extraction_timeout,  # Slightly longer timeout for livestreams
                        )
                        debug_utils.emit_debug(
                            f"FFmpeg result for segment {segment_index}: stdout={len(ffmpeg_result.stdout)}B, stderr={len(ffmpeg_result.stderr)}B",
                            "info",
                            {
                                "stderr": (
                                    ffmpeg_result.stderr
                                    if ffmpeg_result.stderr
                                    else "None"
                                )
                            },
                        )
                    except subprocess.TimeoutExpired:
                        debug_utils.emit_debug(
                            f"FFmpeg process timed out for segment {segment_index}",
                            "warning",
                        )
                        # For sequential extraction, we should pause before trying again
                        time.sleep(
                            2
                        )  # Wait a bit for more stream data to become available
                        consecutive_failures += 1
                        continue  # Skip to next iteration without incrementing position

                    # Check if file was created and has content
                    if (
                        not os.path.exists(segment_file)
                        or os.path.getsize(segment_file) < 1000
                    ):  # Require at least 1KB
                        debug_utils.emit_debug(
                            f"FFmpeg failed to extract segment or file is too small: {segment_file}. File exists: {os.path.exists(segment_file)}, Size: {os.path.getsize(segment_file) if os.path.exists(segment_file) else 0}",
                            "warning",
                        )

                        # Increment failure counter
                        consecutive_failures += 1

                        # If several consecutive failures occur without a recent refresh, force a refresh
                        if (
                            consecutive_failures >= max_consecutive_failures
                            and time.time() - last_refresh_time > 30
                        ):
                            debug_utils.emit_debug(
                                f"Too many consecutive failures ({consecutive_failures}), forcing stream URL refresh",
                                "warning",
                            )
                            last_refresh_time = 0  # Force refresh on next iteration

                        # Wait briefly before retrying
                        time.sleep(0.5)
                        # Continue to next segment with slight progression to avoid getting stuck
                        segment_index += 1
                        start_time += 0.5  # Advance slightly to try a different portion of the stream
                        continue

                    # File was created, now convert to WAV
                    consecutive_failures = 0  # Reset failure counter on success
                    file_size_kb = os.path.getsize(segment_file) / 1024
                    debug_utils.emit_debug(
                        f"Successfully extracted segment {segment_index}, size: {file_size_kb:.2f}KB",
                        "info",
                    )

                    # Convert to WAV format for Whisper API
                    wav_cmd = [
                        "ffmpeg",
                        "-i",
                        segment_file,  # Input file
                        "-ar",
                        "16000",  # 16kHz sample rate for Whisper
                        "-ac",
                        "1",  # Mono audio
                        "-y",  # Overwrite
                        wav_file,  # Output file
                    ]

                    debug_utils.emit_debug(
                        f"Converting segment {segment_index} to WAV format", "info"
                    )

                    # Reduced timeout for WAV conversion
                    wav_result = subprocess.run(
                        wav_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.process_timeout,  # Shorter timeout for WAV conversion
                    )

                    if not os.path.exists(wav_file) or os.path.getsize(wav_file) < 1000:
                        debug_utils.emit_debug(
                            f"Failed to convert segment {segment_index} to WAV format",
                            "warning",
                        )
                        # Try the next segment
                        segment_index += 1
                        start_time += segment_duration
                        continue

                    # Read the WAV file
                    with open(wav_file, "rb") as f:
                        audio_data = f.read()

                    wav_size_kb = len(audio_data) / 1024
                    debug_utils.emit_debug(
                        f"Read WAV segment {segment_index}, size: {wav_size_kb:.2f}KB",
                        "info",
                    )

                    # Apply adaptive chunking
                    chunk_duration, increase_overlap = self.adaptive_chunk_size(
                        audio_data, start_time, base_chunk_duration=CHUNK_DURATION
                    )

                    # Determine overlap based on speech characteristics
                    overlap = OVERLAP_DURATION
                    if increase_overlap:
                        # Cap the overlap to avoid excessive overlapping
                        overlap = min(OVERLAP_DURATION * 1.5, CHUNK_DURATION / 3)
                        debug_utils.emit_debug(
                            f"Increasing overlap to {overlap:.1f}s due to speech characteristics",
                            "info",
                        )

                    # Enhance the audio before processing
                    enhanced_audio = self.enhance_audio(audio_data)
                    debug_utils.emit_debug(
                        f"Enhanced audio chunk {segment_index}, original size: {len(audio_data)} bytes, enhanced size: {len(enhanced_audio)} bytes",
                        "info",
                    )

                    # Process the audio chunk
                    if self.chunk_callback:
                        # Ensure chunks are aligned with 15-second boundaries for consistent display
                        # Calculate a start_time that aligns with multiples of ADVANCE_DURATION (5 seconds)
                        aligned_start_time = (
                            round(start_time / ADVANCE_DURATION) * ADVANCE_DURATION
                        )

                        debug_utils.emit_debug(
                            f"Processing audio chunk: {aligned_start_time:.1f}s - {aligned_start_time+CHUNK_DURATION:.1f}s (overlap: {OVERLAP_DURATION:.1f}s)",
                            "info",
                        )
                        # Use the enhanced audio with properly aligned timestamps
                        self.chunk_callback(
                            enhanced_audio,
                            aligned_start_time,
                            aligned_start_time + CHUNK_DURATION,
                        )
                        debug_utils.emit_debug(
                            f"Audio chunk {segment_index} processed successfully",
                            "info",
                        )

                    # Increment counters - keep these for logical time tracking
                    # even though we're using sequential extraction for the actual FFmpeg command
                    segment_index += 1
                    # Always advance by exactly ADVANCE_DURATION (5 seconds) to maintain alignment
                    start_time += ADVANCE_DURATION  # Exactly 5 seconds for perfect 15-second chunks

                    # Add a delay to ensure we have enough new content in the livestream
                    # This prevents processing chunks faster than new content is created
                    debug_utils.emit_debug(
                        f"Waiting for next {CHUNK_DURATION}-second chunk to become available in livestream...",
                        "info",
                    )
                    for _ in range(
                        5
                    ):  # Break the wait into 1-second increments (reduced from 15 to 5)
                        if stop_event and stop_event.is_set():
                            break
                        time.sleep(1)

                    # Clean up the temporary files
                    try:
                        os.remove(segment_file)
                        os.remove(wav_file)
                    except Exception as e:
                        debug_utils.emit_debug(
                            f"Error cleaning up temp files: {str(e)}", "warning"
                        )

                except subprocess.TimeoutExpired:
                    # This should no longer be reached for FFmpeg timeouts, but kept for other places
                    debug_utils.emit_debug(
                        f"Process timed out for segment {segment_index}",
                        "warning",
                    )
                    # Don't increment start_time for sequential extraction
                    consecutive_failures += 1
                    time.sleep(1)  # Brief pause

                except Exception as e:
                    debug_utils.emit_debug(
                        f"Error processing segment {segment_index}: {str(e)}",
                        "error",
                        {"exception": traceback.format_exc()},
                    )
                    # Don't increment start_time for sequential extraction,
                    # but still increment segment index for logging purposes
                    consecutive_failures += 1
                    segment_index += 1
                    time.sleep(1)  # Brief pause

        except Exception as e:
            debug_utils.emit_debug(
                f"Fatal error in livestream processing: {str(e)}",
                "error",
                {"exception": traceback.format_exc()},
            )

        finally:
            debug_utils.emit_debug("Audio processing thread completed", "info")
            # Clean up temporary directory
            try:
                debug_utils.emit_debug(
                    f"Cleaning up temporary directory: {segments_dir}", "info"
                )
                shutil.rmtree(segments_dir, ignore_errors=True)
            except Exception as e:
                debug_utils.emit_debug(
                    f"Error cleaning up temp files: {str(e)}", "warning"
                )

    def _process_audio_file(self, file_path):
        """Process an audio file by splitting it into chunks."""
        debug_utils.emit_debug(f"Processing audio file: {file_path}")

        if not os.path.exists(file_path):
            debug_utils.emit_debug(f"Audio file not found: {file_path}")
            return

        file_size = os.path.getsize(file_path)
        debug_utils.emit_debug(f"Audio file size: {file_size} bytes")

        if file_size == 0:
            debug_utils.emit_debug("Audio file is empty, skipping processing")
            return

        # Create a temporary directory for chunks
        chunks_dir = os.path.join(self.temp_dir, f"chunks_{int(time.time())}")
        os.makedirs(chunks_dir, exist_ok=True)
        debug_utils.emit_debug(f"Created chunk directory: {chunks_dir}")

        # Use ffmpeg to split the audio file into chunks of specified duration
        cmd = f'ffmpeg -i {file_path} -f segment -segment_time {self.chunk_size - 0.5} -c:a pcm_s16le -ar 16000 -ac 1 -af volume=1.5 {os.path.join(chunks_dir, "chunk_%03d.wav")}'
        debug_utils.emit_debug(f"Executing FFmpeg command: {cmd}")

        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()

        debug_utils.emit_debug(
            f"FFmpeg process completed with return code: {process.returncode}"
        )
        debug_utils.emit_debug(f"FFmpeg stderr:", {"stderr": stderr})

        # Process each chunk
        if process.returncode == 0:
            chunk_files = sorted(
                [
                    f
                    for f in os.listdir(chunks_dir)
                    if f.startswith("chunk_") and f.endswith(".wav")
                ]
            )
            debug_utils.emit_debug(
                f"Found {len(chunk_files)} chunks", {"chunk_files": chunk_files}
            )

            for i, chunk_file in enumerate(chunk_files):
                chunk_path = os.path.join(chunks_dir, chunk_file)
                start_time = i * self.chunk_size
                end_time = (i + 1) * self.chunk_size

                debug_utils.emit_debug(
                    f"Processing chunk: {chunk_file}",
                    {"start_time": start_time, "end_time": end_time},
                )

                # Read the audio data
                with open(chunk_path, "rb") as f:
                    audio_data = f.read()
                    debug_utils.emit_debug(
                        f"Read {len(audio_data)} bytes from chunk file"
                    )

                    # Call the callback function with the chunk data
                    if self.chunk_callback:
                        debug_utils.emit_debug(
                            "Calling callback function with audio chunk"
                        )
                        self.chunk_callback(audio_data, start_time, end_time)
                        debug_utils.emit_debug(
                            f"Received chunk {i+1}: {len(audio_data)} bytes, time {start_time:.2f}s to {end_time:.2f}s"
                        )

        # Clean up
        try:
            if os.path.exists(chunks_dir):
                for filename in os.listdir(chunks_dir):
                    file_path = os.path.join(chunks_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(chunks_dir)
                debug_utils.emit_debug(f"Cleaned up chunk directory: {chunks_dir}")
        except Exception as e:
            debug_utils.emit_debug(f"Error cleaning up chunk directory: {str(e)}")

        debug_utils.emit_debug("Removed temporary files")

    def _download_audio(self, url):
        # Implementation of _download_audio method
        pass

    def _process_audio_data(self, audio_data):
        # Implementation of _process_audio_data method
        pass

    def debug_print(self, message, data=None):
        """Print debug message using the debug_utils module."""
        debug_utils.emit_debug(message, "info", data)

    def stop_extraction(self):
        """Stop the audio extraction process."""
        if self.processing_thread and self.processing_thread.is_alive():
            debug_utils.emit_debug("Stopping audio extraction", "info")
            self.stop_extraction_event.set()
            return True
        return False

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for file_path in self.current_temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    debug_utils.emit_debug(
                        f"Removed temporary file: {file_path}", "info"
                    )
            except Exception as e:
                debug_utils.emit_debug(
                    f"Error removing temporary file {file_path}: {str(e)}",
                    "error",
                    {"error": str(e), "file_path": file_path},
                )

        self.current_temp_files = []

    def enhance_audio(self, audio_data, sample_rate=16000):
        """
        Enhance audio quality for better transcription accuracy.

        Args:
            audio_data (bytes): Raw audio data
            sample_rate (int): Sample rate of the audio

        Returns:
            bytes: Enhanced audio data
        """
        debug_utils.emit_debug(
            f"Enhancing audio quality, original size: {len(audio_data)} bytes", "info"
        )

        try:
            try:
                # Check if pydub is available
                import pydub
                from pydub import AudioSegment
                from pydub.effects import normalize

                pydub_available = True
            except ImportError:
                debug_utils.emit_debug(
                    "pydub library not available. Audio enhancement will be limited.",
                    "warning",
                )
                pydub_available = False

            # Create a temporary file path for working with audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(audio_data)

            enhanced_audio_data = audio_data  # Default to original if enhancement fails

            if pydub_available:
                # Load the audio file with pydub
                audio_segment = AudioSegment.from_file(tmp_file_path)

                # Apply audio enhancements
                # 1. Normalize volume
                enhanced = normalize(audio_segment, headroom=0.1)

                # 2. Boost volume slightly to ensure audibility
                enhanced = enhanced + 3  # Boost by 3dB

                # Save the enhanced audio to a temporary file
                enhanced_path = tmp_file_path + "_enhanced.wav"
                enhanced.export(enhanced_path, format="wav")

                # Read the processed file back as bytes
                with open(enhanced_path, "rb") as f:
                    enhanced_audio_data = f.read()

                # Clean up enhanced audio temp file
                if os.path.exists(enhanced_path):
                    os.remove(enhanced_path)

            # Always try to apply basic numpy-based enhancements if scipy is available
            try:
                # Load as numpy array for further processing
                y, sr = librosa.load(tmp_file_path, sr=sample_rate)

                # Apply mild compression to reduce dynamic range
                # Simple compression by reducing amplitude of peaks
                percentile = np.percentile(np.abs(y), 95)
                mask = np.abs(y) > percentile
                y[mask] = np.sign(y[mask]) * (
                    percentile + (np.abs(y[mask]) - percentile) * 0.5
                )

                # Apply normalization if the audio is too quiet
                if np.max(np.abs(y)) < 0.1:
                    y = y / (np.max(np.abs(y)) + 1e-10) * 0.9

                # Save the final processed audio
                final_path = tmp_file_path + "_final.wav"
                sf.write(final_path, y, sample_rate)

                # Read the processed file back as bytes
                with open(final_path, "rb") as f:
                    enhanced_audio_data = f.read()

                # Clean up final audio temp file
                if os.path.exists(final_path):
                    os.remove(final_path)

            except Exception as e:
                debug_utils.emit_debug(
                    f"Error in numpy-based enhancement: {str(e)}", "warning"
                )
                # If this enhancement fails, we still have the pydub enhanced or original audio

            # Clean up the original temp file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

            debug_utils.emit_debug(
                f"Audio enhancement complete, new size: {len(enhanced_audio_data)} bytes",
                "info",
            )

            return enhanced_audio_data

        except Exception as e:
            debug_utils.emit_debug(
                f"Error enhancing audio: {str(e)}", "error", {"error": str(e)}
            )
            # If enhancement fails, return the original audio
            return audio_data

    def create_multi_format_audio(self, audio_data):
        """
        Create multiple formats of the same audio for better transcription results.

        Args:
            audio_data (bytes): Original audio data

        Returns:
            list: List of audio data in different formats
        """
        debug_utils.emit_debug(
            "Creating multiple audio formats for transcription", "info"
        )

        # Check if we have the necessary libraries
        try:
            import librosa
            import soundfile as sf

            libraries_available = True
        except ImportError:
            debug_utils.emit_debug(
                "Required audio libraries (librosa/soundfile) not available. Using original format only.",
                "warning",
            )
            return [{"data": audio_data, "sample_rate": 16000, "channels": 1}]

        audio_formats = []

        try:
            # Create a temporary WAV file from the original audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(audio_data)

            # Process each format
            for i, format_spec in enumerate(AUDIO_FORMATS):
                sr = format_spec["sample_rate"]
                channels = format_spec["channels"]

                # Load audio with the specific sample rate
                y, _ = librosa.load(tmp_file_path, sr=sr, mono=(channels == 1))

                # Generate a new temporary file for this format
                format_path = f"{tmp_file_path}_format_{i}.wav"
                sf.write(format_path, y, sr)

                # Read the new format file
                with open(format_path, "rb") as f:
                    format_data = f.read()

                audio_formats.append(
                    {"data": format_data, "sample_rate": sr, "channels": channels}
                )

                # Clean up
                if os.path.exists(format_path):
                    os.remove(format_path)

            # Clean up the original temp file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

            debug_utils.emit_debug(
                f"Created {len(audio_formats)} different audio formats", "info"
            )

            return audio_formats

        except Exception as e:
            debug_utils.emit_debug(
                f"Error creating multiple audio formats: {str(e)}",
                "error",
                {"error": str(e)},
            )
            # If creation fails, return a list with just the original audio
            return [{"data": audio_data, "sample_rate": 16000, "channels": 1}]

    def _detect_silence(self, audio_data, threshold=0.02, min_silence_duration=0.5):
        """
        Detect silence in audio to optimize chunk boundaries.

        Args:
            audio_data (bytes): Audio data to analyze
            threshold (float): Amplitude threshold for silence detection
            min_silence_duration (float): Minimum silence duration in seconds

        Returns:
            list: List of silence intervals as (start_time, end_time) tuples
        """
        debug_utils.emit_debug("Detecting silence in audio chunk", "info")

        # Check if we have the necessary libraries
        try:
            import librosa
            import numpy as np

            libraries_available = True
        except ImportError:
            debug_utils.emit_debug(
                "Required audio libraries (librosa/numpy) not available. Cannot detect silence.",
                "warning",
            )
            return []

        try:
            # Create a temporary file for the audio data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(audio_data)

            # Load audio with librosa
            y, sr = librosa.load(tmp_file_path, sr=None)

            # Clean up the temporary file
            os.remove(tmp_file_path)

            # Get amplitude envelope
            amplitude_envelope = np.abs(y)

            # Find regions below threshold
            is_silent = amplitude_envelope < threshold

            # Find silence boundaries
            silent_regions = []
            silent_start = None

            for i, silent in enumerate(is_silent):
                # Start of silence region
                if silent and silent_start is None:
                    silent_start = i / sr
                # End of silence region
                elif not silent and silent_start is not None:
                    silent_end = i / sr
                    duration = silent_end - silent_start
                    if duration >= min_silence_duration:
                        silent_regions.append((silent_start, silent_end))
                    silent_start = None

            # Handle case where the audio ends during silence
            if silent_start is not None:
                silent_end = len(y) / sr
                duration = silent_end - silent_start
                if duration >= min_silence_duration:
                    silent_regions.append((silent_start, silent_end))

            debug_utils.emit_debug(
                f"Detected {len(silent_regions)} silence regions in audio",
                "info",
                {"silence_regions": silent_regions[:5] if silent_regions else []},
            )

            return silent_regions

        except Exception as e:
            debug_utils.emit_debug(
                f"Error detecting silence: {str(e)}", "error", {"error": str(e)}
            )
            return []

    def adaptive_chunk_size(
        self, audio_data, start_time, base_chunk_duration=CHUNK_DURATION
    ):
        """Determine the optimal chunk size and overlap based on audio characteristics.

        This function analyzes the audio to detect silence regions but always
        returns a consistent chunk_duration of CHUNK_DURATION (15 seconds) to
        ensure regular timestamps in the final output.

        Args:
            audio_data (bytes): The audio data
            start_time (float): The start time of the audio
            base_chunk_duration (float): The base duration for chunks (defaults to 15s)

        Returns:
            tuple: (chunk_duration, increase_overlap) - chunk_duration is always 15s,
            and increase_overlap is a boolean indicating if we should increase overlap
        """
        # Detect silence regions for informational purposes only
        silence_regions = self._detect_silence(audio_data)

        debug_utils.emit_debug(
            f"Detected {len(silence_regions)} silence regions in audio",
            "info",
            {"silence_regions": silence_regions},
        )

        # Always use the fixed CHUNK_DURATION (15 seconds) for consistency
        # but determine if we need increased overlap based on silence
        increase_overlap = False

        # If there are silence regions, check if they're at the edges
        if silence_regions:
            # If silence at the end, it might indicate we should increase overlap
            # to ensure we capture continuing speech in the next chunk
            audio_duration = len(audio_data) / (SAMPLE_RATE * 2)  # Approximate duration
            for start, end in silence_regions:
                # If silence is near the end (last 20% of the audio)
                if end > audio_duration * 0.8:
                    increase_overlap = True
                    break

        # Return fixed chunk duration and overlap adjustment flag
        return CHUNK_DURATION, increase_overlap
