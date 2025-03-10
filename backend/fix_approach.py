#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Approach Script for YouTube Livestream Transcriber

This script demonstrates a working approach to transcribing YouTube livestreams
by using yt-dlp to download audio segments first.
"""

import os
import subprocess
import threading
import time
import tempfile
import sys


def main():
    """Main function to demonstrate the working approach."""

    print("YouTube Livestream Transcriber - Fix Approach")
    print("===========================================")

    # Check for YouTube URL argument
    if len(sys.argv) < 2:
        print("Usage: python fix_approach.py [YouTube URL]")
        print(
            "Example: python fix_approach.py https://www.youtube.com/live/vuTY4FDAbpA"
        )
        return

    youtube_url = sys.argv[1]
    print(f"YouTube URL: {youtube_url}")

    # Create a test directory
    temp_dir = tempfile.mkdtemp(prefix="ytfix_")
    print(f"Created temporary directory: {temp_dir}")

    # Step 1: Use yt-dlp to download a short segment of audio
    print("Downloading audio segment using yt-dlp...")
    download_cmd = [
        "yt-dlp",
        "--format",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--postprocessor-args",
        "-ar 16000 -ac 1",
        "--max-filesize",
        "10M",  # Limit to 10MB
        "--output",
        os.path.join(temp_dir, "audio.%(ext)s"),
        youtube_url,
    ]

    try:
        subprocess.run(download_cmd, check=True, text=True, capture_output=True)
        print("Audio segment downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio: {e}")
        print(f"Error output: {e.stderr}")
        return

    # Step 2: Process the downloaded audio into chunks
    audio_file = os.path.join(temp_dir, "audio.wav")
    if not os.path.exists(audio_file):
        print(f"Downloaded audio file not found at: {audio_file}")
        return

    print(f"Downloaded audio file size: {os.path.getsize(audio_file)} bytes")

    # Step 3: Split the audio into chunks using FFmpeg
    print("Splitting audio into chunks...")
    chunk_dir = os.path.join(temp_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        audio_file,
        "-f",
        "segment",
        "-segment_time",
        "5",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        os.path.join(chunk_dir, "chunk_%03d.wav"),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, text=True, capture_output=True)
        print("Audio split into chunks successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error splitting audio: {e}")
        print(f"Error output: {e.stderr}")
        return

    # Step 4: List the created chunk files
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".wav")])
    print(f"\nCreated {len(chunk_files)} chunk files:")
    for chunk_file in chunk_files[:5]:  # Show first 5 files
        file_path = os.path.join(chunk_dir, chunk_file)
        print(f"  {chunk_file} - {os.path.getsize(file_path)} bytes")

    if len(chunk_files) > 5:
        print(f"  ... and {len(chunk_files) - 5} more files")

    print("\nSUCCESS - This demonstrates a working approach!")
    print("Recommendation: Modify the AudioProcessor to use this two-step approach:")
    print("1. Download audio segments using yt-dlp")
    print("2. Process the downloaded segments using FFmpeg")
    print("This avoids the issues with FFmpeg's HLS stream processing.")


if __name__ == "__main__":
    main()
