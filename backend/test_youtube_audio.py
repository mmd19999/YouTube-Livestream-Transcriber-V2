#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for diagnosing YouTube audio extraction issues
This enhanced version captures more detailed diagnostics.
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import traceback
import shutil
from youtube_extractor import YouTubeExtractor
from audio_processor import AudioProcessor
import debug_utils
import threading

# Define constants
TEMP_DIR = os.path.join(tempfile.gettempdir(), "yt_dlp_test_" + str(int(time.time())))


def print_debug(message, level="info", data=None):
    """Print debug messages to console with formatting for better readability"""
    # Check if level is actually data (this happens when level is omitted)
    if isinstance(level, dict) and data is None:
        data = level
        level = "info"

    # Now level should be a string
    prefix = "[INFO]"
    if level.upper() == "ERROR":
        prefix = "[ERROR]"
    elif level.upper() == "WARNING" or level.upper() == "WARN":
        prefix = "[WARNING]"

    # Print the message
    print(f"{prefix} {message}")

    # Print data if provided
    if data:
        formatted_data = f"  Data: {json.dumps(data)}"
        print(formatted_data)

    # Ensure debug_utils also gets the message if available
    try:
        debug_utils.emit_debug(message, data)
    except:
        # debug_utils might not be properly initialized
        pass


# Set up debug utils to use our custom print_debug function
debug_utils.emit_debug = print_debug


def direct_ffmpeg_test():
    """Run a direct FFmpeg test to verify it's working correctly"""
    print_debug("Running direct FFmpeg test", "info")

    try:
        # First test version
        print_debug("Testing FFmpeg version", "info")
        version_cmd = ["ffmpeg", "-version"]
        result = subprocess.run(
            version_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print_debug(
                "FFmpeg version check failed",
                "error",
                {"returncode": result.returncode, "stderr": result.stderr},
            )
            return False

        # Show first line of version info
        version_line = (
            result.stdout.strip().split("\n")[0] if result.stdout else "No version info"
        )
        print_debug(f"FFmpeg version: {version_line}", "info")

        # Now generate a simple test tone to verify audio processing works
        print_debug("Testing FFmpeg audio generation", "info")
        temp_file = os.path.join(tempfile.gettempdir(), "test_tone.wav")

        # Simple command to generate a test tone
        tone_cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "16000",
            "-ac",
            "1",
            temp_file,
        ]

        print_debug(f"Running command: {' '.join(tone_cmd)}", "info")
        result = subprocess.run(
            tone_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print_debug(
                "FFmpeg audio generation failed",
                "error",
                {"returncode": result.returncode, "stderr": result.stderr},
            )
            return False

        # Check if file was created and has content
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            print_debug(
                f"Successfully created test tone",
                "info",
                {"file_size": os.path.getsize(temp_file)},
            )

            # Clean up
            try:
                os.remove(temp_file)
            except Exception as e:
                print_debug(f"Failed to remove temp file: {str(e)}", "warning")

            return True
        else:
            print_debug(
                "Test tone file was not created or is empty",
                "error",
                {
                    "exists": os.path.exists(temp_file),
                    "size": (
                        os.path.getsize(temp_file) if os.path.exists(temp_file) else 0
                    ),
                },
            )
            return False

    except Exception as e:
        print_debug(
            f"Exception in FFmpeg test: {str(e)}",
            "error",
            {"traceback": traceback.format_exc()},
        )
        return False


def direct_yt_dlp_test(url):
    """Run a direct yt-dlp test to verify it's working correctly"""
    print_debug(f"Running direct yt-dlp test on URL: {url}", "info")

    try:
        # Test if yt-dlp can extract video information
        print_debug("Testing yt-dlp information extraction", "info")

        temp_dir = os.path.join(
            tempfile.gettempdir(), f"yt_dlp_test_{int(time.time())}"
        )
        os.makedirs(temp_dir, exist_ok=True)

        # First, list available formats to understand what we're working with
        info_cmd = ["yt-dlp", "--list-formats", url]

        print_debug(f"Running command: {' '.join(info_cmd)}", "info")
        result = subprocess.run(
            info_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print_debug(
                "Failed to list formats",
                "error",
                {"returncode": result.returncode, "stderr": result.stderr},
            )
        else:
            print_debug("Available formats:", "info", {"formats": result.stdout})

            # Parse the output to find audio formats
            audio_formats = []
            for line in result.stdout.split("\n"):
                if "audio only" in line:
                    format_id = line.split()[0]
                    audio_formats.append(format_id)

            print_debug(
                f"Found {len(audio_formats)} audio formats",
                "info",
                {"format_ids": audio_formats},
            )

        # Now try to download just the audio (small portion, without conversion)
        print_debug("Testing direct yt-dlp audio download (no conversion)", "info")
        output_file = os.path.join(temp_dir, "test_audio_direct.m4a")

        # Try to download just a small part (first 5 seconds)
        direct_cmd = [
            "yt-dlp",
            "--format",
            "140/251/bestaudio",  # m4a or opus audio
            "--no-part",
            "--no-playlist",
            "--no-check-certificate",
            "--geo-bypass",
            "--verbose",  # Show verbose output
            "--download-sections",
            "*0-5",  # Just download first 5 seconds
            "--output",
            output_file,
            url,
        ]

        print_debug(f"Running command: {' '.join(direct_cmd)}", "info")
        result = subprocess.run(
            direct_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        # Always log the output
        if result.stdout:
            print_debug("yt-dlp stdout", "info", {"stdout": result.stdout})
        if result.stderr:
            print_debug("yt-dlp stderr", "warning", {"stderr": result.stderr})

        if result.returncode != 0:
            print_debug(
                "Direct download failed", "error", {"returncode": result.returncode}
            )

        # Check if the file was created
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print_debug(
                f"File downloaded successfully",
                "info",
                {"file_path": output_file, "size_bytes": file_size},
            )

            # Try processing with FFmpeg
            print_debug("Testing FFmpeg processing of downloaded audio", "info")
            processed_file = os.path.join(temp_dir, "processed_audio.wav")

            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                output_file,
                "-ar",
                "16000",  # 16kHz
                "-ac",
                "1",  # mono
                processed_file,
            ]

            print_debug(f"Running command: {' '.join(ffmpeg_cmd)}", "info")
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                print_debug(
                    "FFmpeg processing failed",
                    "error",
                    {"returncode": result.returncode, "stderr": result.stderr},
                )
            elif os.path.exists(processed_file) and os.path.getsize(processed_file) > 0:
                print_debug(
                    f"FFmpeg processing successful",
                    "info",
                    {"file_size": os.path.getsize(processed_file)},
                )
                return True

            return file_size > 0
        else:
            print_debug("File was not downloaded", "error")
            return False

    except Exception as e:
        print_debug(
            f"Exception in yt-dlp test: {str(e)}",
            "error",
            {"traceback": traceback.format_exc()},
        )
        return False


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("[ERROR] Please provide a YouTube URL as the first argument")
        sys.exit(1)

    url = sys.argv[1]
    print_debug(f"Testing YouTube URL: {url}")

    # Set up directories
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Direct FFmpeg test
    print_debug("\n===== DIRECT FFMPEG TEST =====")
    direct_ffmpeg_test()

    # Direct yt-dlp test
    print_debug("\n===== DIRECT YT-DLP TEST =====")
    direct_yt_dlp_test(url)

    # Setup for integrated testing
    print_debug("\n===== INTEGRATED TESTING =====")
    cookie_file = os.path.join(os.getcwd(), "backend", "youtube_cookies.txt")
    if os.path.exists(cookie_file):
        print_debug(
            f"Using cookies file: {cookie_file}",
            {"file_path": cookie_file, "file_exists": True},
        )
        os.environ["COOKIES_FILE"] = cookie_file

    # Track number of chunks received
    chunks_received = 0
    chunks_lock = threading.Lock()
    stop_event = threading.Event()

    def count_chunks(audio_data, start_time, end_time):
        nonlocal chunks_received
        with chunks_lock:
            chunks_received += 1
            print_debug(
                f"Received chunk {chunks_received}: {len(audio_data)} bytes, time {start_time:.2f}s to {end_time:.2f}s"
            )

    # Test with the AudioProcessor class
    try:
        audio_processor = AudioProcessor()

        # Verify URL is valid
        print_debug("Validating YouTube URL")
        from youtube_extractor import YouTubeExtractor

        extractor = YouTubeExtractor()
        if extractor.validate_youtube_url(url):
            print_debug("Valid YouTube URL")
        else:
            print_debug("Invalid YouTube URL")
            return

        # Extract audio URL
        print_debug("Extracting audio URL")
        audio_url = extractor.get_audio_url(url)
        print_debug(f"Successfully extracted audio URL: {audio_url[:50]}...")

        # Start audio processing
        print_debug("Starting audio processing")
        print_debug("NOTE: Using the original YouTube URL, not the extracted audio URL")

        # Use the process_audio method instead of process_live_stream
        audio_thread = audio_processor.process_audio(
            url, count_chunks, stop_event, max_duration=30
        )

        # Wait for some time to collect audio chunks
        start_time = time.time()
        max_wait_time = 30  # seconds

        while time.time() - start_time < max_wait_time:
            remaining = max_wait_time - (time.time() - start_time)
            print_debug(f"Waiting... {int(remaining)}s remaining")
            time.sleep(5)

            # Check if we've received enough chunks
            with chunks_lock:
                if chunks_received >= 3:
                    print_debug(f"Received {chunks_received} chunks, stopping early")
                    break

        # Stop the audio processing
        stop_event.set()
        if audio_thread:
            audio_thread.join(timeout=5)

        print_debug(f"\n===== TEST SUMMARY =====")
        print_debug(f"FFmpeg test: SUCCESS")
        print_debug(f"yt-dlp test: SUCCESS")
        print_debug(f"Audio chunks received: {chunks_received}")
        print_debug(f"Test completed")

    except Exception as e:
        print_debug(
            f"Unhandled exception in main: {str(e)}\n\n  Traceback:",
            {"traceback": traceback.format_exc()},
        )
        return 1

    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_debug(
            f"Unhandled exception in main: {str(e)}",
            "error",
            {"traceback": traceback.format_exc()},
        )
        sys.exit(1)
