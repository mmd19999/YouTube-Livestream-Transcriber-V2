#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import logging
import subprocess
from pathlib import Path
import yt_dlp
import debug_utils
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("youtube_extractor")

# Constants
COOKIES_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "youtube_cookies.txt"
)
AUDIO_FORMAT = "bestaudio/best"
MAX_RETRIES = 3


class YouTubeExtractor:
    """Class to handle YouTube livestream extraction using yt-dlp."""

    def __init__(self, cookies_file=COOKIES_FILE):
        """Initialize the YouTube extractor.

        Args:
            cookies_file (str): Path to the cookies file for authentication.
        """
        self.cookies_file = cookies_file
        self._validate_cookies_file()

    def _validate_cookies_file(self):
        """Validate that the cookies file exists."""
        if not os.path.exists(self.cookies_file):
            logger.error(f"Cookies file not found: {self.cookies_file}")
            debug_utils.emit_debug(
                f"Cookies file not found: {self.cookies_file}",
                "error",
                {"file_path": self.cookies_file},
            )
            raise FileNotFoundError(f"Cookies file not found: {self.cookies_file}")
        logger.info(f"Using cookies file: {self.cookies_file}")
        debug_utils.emit_debug(
            f"Using cookies file: {self.cookies_file}",
            "info",
            {"file_path": self.cookies_file, "file_exists": True},
        )

    @debug_utils.debug_api_call
    def validate_youtube_url(self, url):
        """Validate if the URL is a valid YouTube URL.

        Args:
            url (str): The URL to validate.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        # YouTube URL patterns
        youtube_patterns = [
            r"^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$",
            r"^(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+(&\S*)?$",
            r"^(https?://)?(www\.)?youtu\.be/[\w-]+(\?\S*)?$",
            r"^(https?://)?(www\.)?youtube\.com/live/[\w-]+(\?\S*)?$",
            r"^(https?://)?(www\.)?youtube\.com/channel/[\w-]+(/live)?(\?\S*)?$",
        ]

        # Check if the URL matches any of the patterns
        for pattern in youtube_patterns:
            if re.match(pattern, url):
                return True

        return False

    @debug_utils.debug_api_call
    def get_stream_info(self, url):
        """Get information about a YouTube livestream.

        Args:
            url (str): The YouTube URL.

        Returns:
            dict: Information about the livestream.
        """
        logger.info(f"Getting stream info for: {url}")

        # Configure yt-dlp options
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "cookiefile": self.cookies_file,
            "skip_download": True,
            "format": AUDIO_FORMAT,
        }

        try:
            # Extract info using yt-dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Check if the video is a livestream
                is_live = info.get("is_live", False)

                # Get duration (0 for livestreams)
                duration = info.get("duration", 0)

                # Get title and channel
                title = info.get("title", "Unknown Title")
                channel = info.get("uploader", "Unknown Channel")

                # Create stream info dictionary
                stream_info = {
                    "title": title,
                    "channel": channel,
                    "is_live": is_live,
                    "duration": duration,
                    "video_id": info.get("id", ""),
                    "url": url,
                }

                logger.info(f"Stream info: {json.dumps(stream_info)}")
                return stream_info

        except Exception as e:
            logger.error(f"Error getting stream info: {str(e)}")
            raise

    @debug_utils.debug_api_call
    def get_audio_url(self, url):
        """Get the audio URL for a YouTube video or livestream.

        Args:
            url (str): YouTube video or livestream URL

        Returns:
            str: Direct audio URL for the stream, or None if extraction fails
        """
        try:
            debug_utils.emit_debug(f"Extracting audio URL for: {url}", "info")

            # Configure yt-dlp options
            ydl_opts = {
                "format": AUDIO_FORMAT,
                "cookiefile": self.cookies_file,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "skip_download": True,
                "extract_flat": False,
            }

            # Extract audio URL using yt-dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if not info:
                    debug_utils.emit_debug(
                        f"Failed to extract info for URL: {url}", "error"
                    )
                    return None

                # Get the best audio format
                formats = info.get("formats", [])
                if not formats:
                    debug_utils.emit_debug(
                        "No formats found in extracted info", "error"
                    )
                    return None

                # Find the best audio format
                audio_formats = [f for f in formats if f.get("acodec") != "none"]
                if not audio_formats:
                    debug_utils.emit_debug("No audio formats found", "error")
                    return None

                # Filter out formats with None values and add debug logging
                audio_formats = [f for f in audio_formats if f is not None]
                debug_utils.emit_debug(
                    f"Found {len(audio_formats)} audio formats", "info"
                )

                # Log format details for debugging
                format_info = [
                    {
                        "format_id": f.get("format_id", "unknown"),
                        "abr": f.get("abr", "unknown"),
                        "acodec": f.get("acodec", "unknown"),
                        "ext": f.get("ext", "unknown"),
                    }
                    for f in audio_formats[:3]
                ]  # Log first 3 formats to avoid excessive logging
                debug_utils.emit_debug(f"Sample audio formats: {format_info}", "info")

                try:
                    # Sort by bitrate (highest first) with safe handling of None values
                    audio_formats.sort(
                        key=lambda x: (
                            x.get("abr", 0) if x.get("abr") is not None else 0
                        ),
                        reverse=True,
                    )

                    if not audio_formats:
                        debug_utils.emit_debug(
                            "No valid audio formats after filtering", "error"
                        )
                        return None

                    best_audio = audio_formats[0]

                    debug_utils.emit_debug(
                        f"Selected audio format: {best_audio.get('format_id')}",
                        "info",
                        {
                            "format_id": best_audio.get("format_id"),
                            "bitrate": best_audio.get("abr", 0),
                            "ext": best_audio.get("ext"),
                        },
                    )
                except Exception as e:
                    debug_utils.emit_debug(
                        f"Error sorting audio formats: {str(e)}",
                        "error",
                        {"error": str(e), "traceback": traceback.format_exc()},
                    )

                    # Fallback: try to get any audio format without sorting
                    if audio_formats:
                        best_audio = audio_formats[0]
                        debug_utils.emit_debug(
                            f"Using fallback audio format: {best_audio.get('format_id')}",
                            "warn",
                        )
                    else:
                        debug_utils.emit_debug(
                            "No audio formats available for fallback", "error"
                        )
                        return None

                # Return the direct audio URL
                audio_url = best_audio.get("url")
                if not audio_url:
                    debug_utils.emit_debug("No URL found in best audio format", "error")
                    return None

                return audio_url

        except Exception as e:
            debug_utils.emit_debug(
                f"Error extracting audio URL: {str(e)}",
                "error",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return None

    # Additional methods can be added here for future functionality


# Example usage
if __name__ == "__main__":
    # Example YouTube URL
    test_url = "https://www.youtube.com/watch?v=jfKfPfyJRdk"  # Lo-Fi Hip Hop Radio

    extractor = YouTubeExtractor()

    # Validate URL
    is_valid = extractor.validate_youtube_url(test_url)
    print(f"Is valid URL: {is_valid}")

    if is_valid:
        # Get stream info
        info = extractor.get_stream_info(test_url)
        print(f"Title: {info['title']}")
        print(f"Channel: {info['channel']}")
        print(f"Is Live: {info['is_live']}")

        try:
            # Get audio URL
            audio_url = extractor.get_audio_url(test_url)
            if audio_url:
                print(
                    f"Audio URL: {audio_url[:100]}..."
                )  # Print first 100 chars of URL
            else:
                print("No audio URL found")
        except Exception as e:
            print(f"Error getting audio URL: {str(e)}")
