#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FFmpeg YouTube Debug Script for YouTube Livestream Transcriber

This is a standalone script to test FFmpeg with YouTube URLs.
"""

import os
import subprocess
import threading
import time
import tempfile
import sys


def main():
    """Main function to test FFmpeg with YouTube URLs."""

    print("FFmpeg YouTube Debug Script")
    print("=========================")

    # Check for YouTube URL argument
    if len(sys.argv) < 2:
        print("Usage: python debug_youtube.py [YouTube URL]")
        print(
            "Example: python debug_youtube.py https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )
        return

    youtube_url = sys.argv[1]
    print(f"YouTube URL: {youtube_url}")

    # Create a test directory
    temp_dir = tempfile.mkdtemp(prefix="youtube_test_")
    print(f"Created temporary directory: {temp_dir}")

    # Extract audio URL using yt-dlp (simpler approach)
    print("Extracting audio URL using yt-dlp...")
    try:
        yt_dlp_cmd = [
            "yt-dlp",
            "-f",
            "bestaudio",
            "-g",  # Just get the URL
            youtube_url,
        ]
        audio_url = subprocess.check_output(yt_dlp_cmd, text=True).strip()
        print(f"Extracted audio URL: {audio_url[:100]}...")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio URL: {e}")
        return
    except FileNotFoundError:
        print("Error: yt-dlp not found. Please install it with 'pip install yt-dlp'")
        return

    # FFmpeg command with reconnection options
    ffmpeg_cmd = [
        "ffmpeg",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "5",
        "-i",
        audio_url,
        "-f",
        "segment",
        "-segment_time",
        "5",
        "-segment_format",
        "wav",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        os.path.join(temp_dir, "chunk_%03d.wav"),
    ]

    print(
        f"FFmpeg command (truncated): ffmpeg ... -i [URL] ... {os.path.join(temp_dir, 'chunk_%03d.wav')}"
    )

    # Start FFmpeg process with full output capture
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    print("FFmpeg process started")

    # Start threads to monitor FFmpeg output
    stderr_lines = []

    def monitor_stderr():
        print("Stderr monitor thread started")
        for line in iter(process.stderr.readline, ""):
            stderr_lines.append(line.strip())
            print(f"FFmpeg: {line.strip()}")

    # Start monitoring thread (focusing on stderr since that's where FFmpeg writes)
    stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
    stderr_thread.start()

    print("Monitoring thread started")
    print("Waiting for FFmpeg to process (max 30 seconds)...")

    # Monitor for new files
    start_time = time.time()
    last_chunk_time = start_time
    chunk_count = 0
    timeout = 30  # seconds

    while time.time() - start_time < timeout:
        # Check if FFmpeg process is still running
        if process.poll() is not None:
            print(f"FFmpeg process terminated with exit code: {process.returncode}")
            break

        # Check for new chunk files
        chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".wav")])
        if len(chunk_files) > chunk_count:
            new_chunks = len(chunk_files) - chunk_count
            print(f"Found {new_chunks} new chunk files (total: {len(chunk_files)})")
            chunk_count = len(chunk_files)
            last_chunk_time = time.time()

        # Check for timeout (no new chunks for a while)
        if chunk_count > 0 and time.time() - last_chunk_time > 10:
            print(f"No new chunks received for 10 seconds, stopping")
            break

        time.sleep(1)

    # Terminate FFmpeg process
    if process.poll() is None:
        print("Terminating FFmpeg process")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("FFmpeg process didn't terminate, killing it")
            process.kill()

    # Wait a bit for monitoring thread to finish
    time.sleep(2)

    # Check for created files
    chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".wav")])
    print(f"\nCreated {len(chunk_files)} chunk files:")
    for chunk_file in chunk_files[:5]:  # Show first 5 files
        file_path = os.path.join(temp_dir, chunk_file)
        print(f"  {chunk_file} - {os.path.getsize(file_path)} bytes")

    if len(chunk_files) > 5:
        print(f"  ... and {len(chunk_files) - 5} more files")

    # Check if we captured any output
    print(f"\nCaptured {len(stderr_lines)} stderr lines")

    if not stderr_lines:
        print(
            "\nNo output captured! This suggests a problem with the output capture mechanism."
        )

    print("\nDONE")


if __name__ == "__main__":
    main()
