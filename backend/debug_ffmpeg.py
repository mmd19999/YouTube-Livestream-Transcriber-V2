#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FFmpeg Debug Script for YouTube Livestream Transcriber

This is a standalone script to test FFmpeg output capture.
"""

import os
import subprocess
import threading
import time
import tempfile


def main():
    """Main function to test FFmpeg output capture."""

    print("FFmpeg Debug Script")
    print("==================")

    # Create a test directory
    temp_dir = tempfile.mkdtemp(prefix="ffmpeg_test_")
    print(f"Created temporary directory: {temp_dir}")

    # Simple FFmpeg command to generate a test tone
    ffmpeg_cmd = [
        "ffmpeg",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=5",
        "-f",
        "segment",
        "-segment_time",
        "1",
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

    print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")

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
    stdout_lines = []
    stderr_lines = []

    def monitor_stdout():
        print("Stdout monitor thread started")
        for line in iter(process.stdout.readline, ""):
            stdout_lines.append(line.strip())
            print(f"FFmpeg stdout: {line.strip()}")

    def monitor_stderr():
        print("Stderr monitor thread started")
        for line in iter(process.stderr.readline, ""):
            stderr_lines.append(line.strip())
            print(f"FFmpeg stderr: {line.strip()}")

    # Start monitoring threads
    stdout_thread = threading.Thread(target=monitor_stdout, daemon=True)
    stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)

    stdout_thread.start()
    stderr_thread.start()

    print("Monitoring threads started")

    # Wait for FFmpeg to finish
    try:
        process.wait(timeout=10)
        print(f"FFmpeg process completed with exit code: {process.returncode}")
    except subprocess.TimeoutExpired:
        print("FFmpeg process timeout - terminating")
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print("FFmpeg process killed")
            process.kill()

    # Wait a bit for monitoring threads to finish
    time.sleep(2)

    # Check for created files
    chunk_files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".wav")])
    print(f"\nCreated {len(chunk_files)} chunk files:")
    for chunk_file in chunk_files:
        file_path = os.path.join(temp_dir, chunk_file)
        print(f"  {chunk_file} - {os.path.getsize(file_path)} bytes")

    # Check if we captured any output
    print(
        f"\nCaptured {len(stdout_lines)} stdout lines and {len(stderr_lines)} stderr lines"
    )

    if not stdout_lines and not stderr_lines:
        print(
            "\nNo output captured! This suggests a problem with the output capture mechanism."
        )

    print("\nDONE")


if __name__ == "__main__":
    main()
