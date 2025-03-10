#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transcription Module for YouTube Livestream Transcriber

This module handles audio transcription using OpenAI's Whisper API,
including temporary file management, API rate limiting, and retry logic.
"""

import os
import time
import tempfile
import threading
import logging
from queue import Queue, Empty
import openai
import debug_utils
from pathlib import Path
import traceback
import numpy as np
import json
import requests
from scipy.io import wavfile
import re
import hashlib
from collections import deque
import soundfile as sf
import scipy.signal as signal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("transcription")

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
WHISPER_MODEL = "whisper-1"
LLM_MODEL = (
    "gpt-3.5-turbo"  # Changed from gpt-4o to ensure compatibility with API 0.27.0
)
TRANSCRIPTION_CONTEXT_WINDOW = 500  # Number of words to keep for context
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score to accept a word
API_RATE_LIMIT = 0.5  # Minimum seconds between API calls to avoid rate limiting

# LLM prompt for overlap detection and resolution
LLM_OVERLAP_PROMPT = """
You are an AI assistant specialized in analyzing overlapping audio transcriptions. Your ONLY task is to extract the unique new content.

PREVIOUS TRANSCRIPTION:
Start Time: {prev_start_time}s
End Time: {prev_end_time}s
Text: {prev_text}

CURRENT TRANSCRIPTION:
Start Time: {curr_start_time}s
End Time: {curr_end_time}s
Text: {curr_text}

INSTRUCTIONS:
1. Compare the PREVIOUS and CURRENT transcriptions word by word.
2. The CURRENT TRANSCRIPTION likely starts with content that duplicates the end of the PREVIOUS TRANSCRIPTION.
3. Identify THE EXACT POINT where the duplication ends and new content begins.
4. Extract ONLY the unique new content that appears after that point.
5. Your response must contain ONLY the unique new text, with no comments or metadata.
6. If there is NO overlap (completely different content), output the entire CURRENT TRANSCRIPTION text unchanged.
7. If the CURRENT TRANSCRIPTION is entirely contained within the PREVIOUS TRANSCRIPTION with no new content, output exactly "<<NO_NEW_CONTENT>>" and nothing else.

OUTPUT RULES:
- Do NOT output ANY duplicate content that appears in both transcriptions
- Do NOT include any statements like "The unique new content is:"
- Do NOT include timestamps or any other metadata
- Do NOT include the overlapping/duplicate content that appears in both transcriptions
- ONLY output the unique new content, nothing else
- If uncertain, err on the side of removing more potential duplicates

EXAMPLES:
PREVIOUS: "My mother says that when I am an adult, I can die."
CURRENT: "My mother says that when I am an adult, I can dye my hair green."
OUTPUT: "dye my hair green."

PREVIOUS: "I tried to tell her that all my friends were doing it"
CURRENT: "I tried to tell her that all my friends were doing it, but she didn't want to listen."
OUTPUT: ", but she didn't want to listen."

PREVIOUS: "I tried to tell her that all my friends were doing it, but she didn't want to listen."
CURRENT: "I tried to tell her that all my friends were doing it, but she didn't want to listen."
OUTPUT: "<<NO_NEW_CONTENT>>"
"""

# Track last API call time
_last_api_call_time = 0


class TranscriptionMemory:
    """Class to maintain context and history of transcriptions for improved accuracy."""

    def __init__(self):
        """Initialize the transcription memory."""
        self.full_history = []  # Complete transcript history
        self.context_window = []  # Recent transcript for context
        self.context_size = 1000  # Limit history size to prevent memory leaks

        # Configure history size via environment variable or use increased default
        # 2000 segments at ~3-5s each = 100-166 minutes of audio, which is reasonable
        default_size = 2000
        env_size = os.environ.get("TRANSCRIPTION_HISTORY_SIZE")
        self.max_history_size = (
            int(env_size) if env_size and env_size.isdigit() else default_size
        )

        debug_utils.emit_debug(
            f"Transcription history size set to {self.max_history_size} segments",
            "info",
        )

        self.lock = threading.RLock()  # Thread safety for memory updates

        # Debug counters
        self.total_words = 0
        self.merged_segments = 0

        # Track last transcription for overlap calculation
        self.last_transcription = None

        debug_utils.emit_debug("TranscriptionMemory initialized", "info")

    def add_transcription(self, transcription):
        """
        Add a transcription to the memory and update context.

        Args:
            transcription (dict): Transcription dict with text and metadata

        Returns:
            dict: The transcription object (possibly modified)
        """
        with self.lock:
            # Skip completely empty transcriptions
            if not transcription or not transcription.get("text"):
                return transcription

            # Clean up any punctuation or formatting issues
            transcription["text"] = self._clean_text(transcription["text"])

            # Skip segments containing LLM instructions that leaked into output
            if (
                "The unique new content in the CURRENT TRANSCRIPTION"
                in transcription["text"]
            ):
                # Filter out LLM instructions from the text
                transcription["text"] = (
                    transcription["text"]
                    .split("The unique new content in the CURRENT TRANSCRIPTION")[0]
                    .strip()
                )
                if not transcription["text"]:
                    return transcription

            if "<<NO_NEW_CONTENT>>" in transcription["text"]:
                # Don't add segments marked as no new content
                transcription["text"] = (
                    transcription["text"].replace("<<NO_NEW_CONTENT>>", "").strip()
                )
                if not transcription["text"]:
                    return transcription

            # Get the words from the text
            transcription_text = transcription.get("text", "").strip()
            words = transcription_text.split()

            # Skip if no actual words (empty after stripping/filtering)
            if not words:
                return transcription

            # Check for duplicates before adding to history
            # Only perform this check for non-merged segments to avoid false positives
            if not transcription.get("is_merged", False) and self.full_history:
                # Check the last few segments for substantial similarity
                for segment in self.full_history[-3:]:
                    segment_text = segment.get("text", "").strip()
                    # If this is substantially the same as an existing segment, don't add it
                    if self._texts_are_substantially_identical(
                        transcription_text, segment_text
                    ):
                        debug_utils.emit_debug(
                            f"Skipping substantially identical segment: '{transcription_text[:50]}...'",
                            "info",
                        )
                        # Return the existing segment instead
                        return segment.copy()

            # Create a unique ID for this segment based on content and timing
            segment_id = hash(
                f"{transcription.get('start_time')}-{transcription.get('end_time')}-{transcription_text[:20]}"
            )
            transcription["segment_id"] = segment_id

            # Add to full history
            self.full_history.append(transcription.copy())
            self.total_words += len(words)

            # Limit history size to prevent memory leaks
            if len(self.full_history) > self.max_history_size:
                self.full_history = self.full_history[-self.max_history_size :]

            # Process the words for the context window
            merged_text = transcription_text

            # Handle context window updates more intelligently

            # For non-merged segments, just add the words normally
            if not transcription.get("is_merged", False):
                # Update the context window (standard behavior)
                self.context_window.extend(words)
                if len(self.context_window) > self.context_size:
                    self.context_window = self.context_window[-self.context_size :]
            else:
                # For merged segments, we need to be more careful to avoid duplication
                # Since the merged segment already contains the context, we should
                # replace the context window rather than extending it

                # Determine how much of the context window to preserve
                # If our context window is getting large, we'll keep some of it
                # to maintain continuity, otherwise reset it completely
                if len(self.context_window) > self.context_size // 2:
                    # Keep a small portion of the old context to maintain some history
                    retain_count = min(self.context_size // 4, len(self.context_window))
                    self.context_window = self.context_window[-retain_count:]
                else:
                    # Context window is small, just clear it
                    self.context_window = []

                # Add the words from the merged text - this becomes our new context
                self.context_window.extend(words)

                # Ensure we stay within size limits
                if len(self.context_window) > self.context_size:
                    self.context_window = self.context_window[-self.context_size :]

            debug_utils.emit_debug(
                f"Added transcription to memory. Total words: {self.total_words}",
                "info",
                {
                    "text_length": len(transcription["text"]),
                    "context_window_size": len(self.context_window),
                    "total_history_segments": len(self.full_history),
                    "is_merged": transcription.get("is_merged", False),
                },
            )

            return transcription

    def get_context(self, word_count=None):
        """
        Get the recent transcription context.

        Args:
            word_count (int, optional): Number of words to return. If None, returns all available context.

        Returns:
            str: The context text
        """
        with self.lock:
            if word_count and word_count < len(self.context_window):
                context = self.context_window[-word_count:]
            else:
                context = self.context_window

            return " ".join(context)

    def merge_transcription(self, transcription):
        """
        Merge a new transcription with the existing context using LLM-based overlap resolution.

        Args:
            transcription (dict): Dictionary containing the transcription text and metadata

        Returns:
            dict: Dictionary with the merged transcription text and metadata
        """
        # Track execution time
        start_time = time.time()

        # Use the LLM-based overlap resolution approach
        result = self.resolve_overlap_with_llm(transcription)

        # Log execution time
        execution_time = time.time() - start_time
        debug_utils.emit_debug(
            f"LLM-based merge operation completed in {execution_time:.3f}s",
            "info",
            {
                "execution_time": execution_time,
                "merged_segments": self.merged_segments,
                "word_count": len(result.get("text", "").split()),
            },
        )

        return result

    def get_full_transcript(self):
        """
        Get the complete transcript history.

        Returns:
            str: The complete transcript text
        """
        with self.lock:
            return " ".join([item["text"] for item in self.full_history])

    def resolve_overlap_with_llm(self, transcription):
        """
        Use LLM to analyze and resolve overlap between the current transcription and the previous one.

        Args:
            transcription (dict): The current transcription segment

        Returns:
            dict: Modified transcription with resolved overlap
        """
        # Guard against empty transcriptions or when there's no previous transcription
        if (
            not transcription
            or "text" not in transcription
            or not transcription["text"].strip()
        ):
            debug_utils.emit_debug("Empty transcription, nothing to resolve", "warning")
            return transcription

        if not self.last_transcription:
            debug_utils.emit_debug("No previous transcription, using as is", "info")
            self.last_transcription = transcription
            return self.add_transcription(transcription)

        # Get timing information
        prev_start_time = self.last_transcription.get("start_time", 0)
        prev_end_time = self.last_transcription.get("end_time", 0)
        prev_text = self.last_transcription.get("text", "").strip()

        curr_start_time = transcription.get("start_time", 0)
        curr_end_time = transcription.get("end_time", 0)
        curr_text = transcription.get("text", "").strip()

        # Clean up text - remove any double periods or other punctuation issues
        prev_text = self._clean_text(prev_text)
        curr_text = self._clean_text(curr_text)

        # Check if texts are substantially identical (ignoring small differences)
        if self._texts_are_substantially_identical(prev_text, curr_text):
            debug_utils.emit_debug(
                "Texts are substantially identical, marking as no new content", "info"
            )
            # Return a copy of the transcription marked as merged with the existing context
            merged_result = transcription.copy()
            merged_result["text"] = prev_text
            merged_result["is_merged"] = True
            # Use current chunk times to maintain consistent 15-second segments
            merged_result["start_time"] = curr_start_time
            merged_result["end_time"] = curr_end_time

            self.merged_segments += 1
            self.last_transcription = merged_result
            return self.add_transcription(merged_result)

        # Verify that there is actually an overlap in time
        if curr_start_time >= prev_end_time:
            debug_utils.emit_debug("No time overlap, using full transcription", "info")
            self.last_transcription = transcription
            return self.add_transcription(transcription)

        debug_utils.emit_debug(
            f"Resolving overlap with LLM. Prev: {prev_start_time:.1f}s-{prev_end_time:.1f}s, Curr: {curr_start_time:.1f}s-{curr_end_time:.1f}s",
            "info",
            {
                "prev_text": prev_text[:50] + ("..." if len(prev_text) > 50 else ""),
                "curr_text": curr_text[:50] + ("..." if len(curr_text) > 50 else ""),
            },
        )

        # Format the prompt with the actual transcription data
        prompt = LLM_OVERLAP_PROMPT.format(
            prev_start_time=prev_start_time,
            prev_end_time=prev_end_time,
            prev_text=prev_text,
            curr_start_time=curr_start_time,
            curr_end_time=curr_end_time,
            curr_text=curr_text,
        )

        try:
            # Call OpenAI API using the 0.27.0 version format
            debug_utils.emit_debug(
                f"Calling OpenAI API with model {LLM_MODEL} for overlap resolution",
                "llm",
                {"prev_length": len(prev_text), "curr_length": len(curr_text)},
            )

            # Implement simple rate limiting
            global _last_api_call_time
            time_since_last_call = time.time() - _last_api_call_time
            if time_since_last_call < API_RATE_LIMIT:
                sleep_time = API_RATE_LIMIT - time_since_last_call
                debug_utils.emit_debug(
                    f"Rate limiting: Sleeping for {sleep_time:.2f}s before API call",
                    "llm",
                )
                time.sleep(sleep_time)

            # Update last API call time
            _last_api_call_time = time.time()

            response = openai.ChatCompletion.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized AI for analyzing transcript overlaps.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Use deterministic output for consistent results
                max_tokens=1024,
            )

            # Extract the response content
            llm_output = response["choices"][0]["message"]["content"].strip()

            debug_utils.emit_debug(
                f"LLM completed overlap analysis. Output length: {len(llm_output)}",
                "llm",
                {
                    "output_sample": llm_output[:50]
                    + ("..." if len(llm_output) > 50 else ""),
                    "full_output": llm_output,
                },
            )

            # Handle the "no new content" case
            if llm_output == "<<NO_NEW_CONTENT>>":
                debug_utils.emit_debug("LLM detected no new content", "info")
                # Return a copy of the transcription marked as merged with the existing context
                merged_result = transcription.copy()

                # IMPORTANT: Keep original current chunk timestamps
                # This ensures we maintain proper 15-second duration in output
                merged_result["text"] = prev_text
                merged_result["is_merged"] = True
                # Use current chunk times instead of extending
                merged_result["start_time"] = curr_start_time
                merged_result["end_time"] = curr_end_time

                self.merged_segments += 1
                self.last_transcription = merged_result
                return self.add_transcription(merged_result)

            # Check if llm_output is very short - might be a truncated/incorrect result
            if len(llm_output.split()) < 2 and len(curr_text.split()) > 5:
                debug_utils.emit_debug(
                    f"LLM returned very short output '{llm_output}' for longer input, possible error",
                    "warning",
                )
                # Fall back to simple overlap detection
                unique_content = self._simple_overlap_detection(prev_text, curr_text)
                # Only use this if we actually found something
                if len(unique_content.split()) >= 2:
                    debug_utils.emit_debug(
                        f"Using simple overlap detection instead, found {len(unique_content.split())} words",
                        "info",
                    )
                    llm_output = unique_content

            # Create the new transcription with the unique content
            unique_content = llm_output

            # If unique content is empty or extremely short, use fallback logic
            if len(unique_content.split()) < 2:
                debug_utils.emit_debug(
                    "LLM returned very little unique content, checking if full overlap",
                    "warning",
                )
                # If times overlap significantly, treat as duplicate
                time_overlap = max(0, prev_end_time - curr_start_time)
                total_time = curr_end_time - curr_start_time
                if total_time > 0 and (time_overlap / total_time) > 0.8:
                    debug_utils.emit_debug(
                        "High time overlap, treating as duplicate", "info"
                    )
                    merged_result = transcription.copy()
                    # Use full current chunk duration
                    merged_result["text"] = prev_text
                    merged_result["is_merged"] = True
                    merged_result["start_time"] = curr_start_time
                    merged_result["end_time"] = curr_end_time
                    self.merged_segments += 1
                    self.last_transcription = merged_result
                    return self.add_transcription(merged_result)

            # For new content, create a clean merged result
            # This helps avoid embedding the "The unique new content..." text
            merged_text = unique_content

            # Create merged transcription object with a single new segment for just the unique content
            merged_transcription = {
                "text": merged_text,
                "start_time": curr_start_time,  # Use current start time for the new unique segment
                "end_time": curr_end_time,  # Maintain 15-second chunk duration
                "pass": transcription.get("pass", 1),
                "timestamp": time.time(),
                "is_merged": True,
            }

            # Mark the original transcription as having been merged
            transcription["was_merged"] = True

            # Update the last transcription
            self.last_transcription = merged_transcription

            debug_utils.emit_debug(
                f"LLM-based merge completed. Result length: {len(merged_text.split())} words",
                "info",
            )

            # Add the merged transcription to memory
            return self.add_transcription(merged_transcription)

        except openai.error.RateLimitError as e:
            debug_utils.emit_debug(
                f"OpenAI API rate limit exceeded: {str(e)}",
                "error",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            # Fallback to using the full current transcription
            self.last_transcription = transcription
            return self.add_transcription(transcription)
        except openai.error.APIError as e:
            debug_utils.emit_debug(
                f"OpenAI API error: {str(e)}",
                "error",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            # Fallback to using the full current transcription
            self.last_transcription = transcription
            return self.add_transcription(transcription)
        except openai.error.Timeout as e:
            debug_utils.emit_debug(
                f"OpenAI API timeout: {str(e)}",
                "error",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            # Fallback to using the full current transcription
            self.last_transcription = transcription
            return self.add_transcription(transcription)
        except Exception as e:
            debug_utils.emit_debug(
                f"Error during LLM overlap resolution: {str(e)}",
                "error",
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            # Fallback to using the full current transcription
            self.last_transcription = transcription
            return self.add_transcription(transcription)

    def _texts_are_substantially_identical(
        self, text1, text2, similarity_threshold=0.85
    ):
        """
        Check if two texts are substantially identical (more efficient than LLM check).

        Args:
            text1 (str): First text to compare
            text2 (str): Second text to compare
            similarity_threshold (float): Threshold for considering texts as identical

        Returns:
            bool: True if texts are substantially identical
        """
        # If one of the texts is empty, they can't be identical
        if not text1 or not text2:
            return False

        # Convert to lowercase for better comparison
        text1 = text1.lower()
        text2 = text2.lower()

        # If exactly the same, return True immediately
        if text1 == text2:
            return True

        # Get word sets
        words1 = set(text1.split())
        words2 = set(text2.split())

        # If one is significantly longer than the other, they're not substantially identical
        if len(words1) > 2 * len(words2) or len(words2) > 2 * len(words1):
            return False

        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return False

        similarity = intersection / union

        return similarity >= similarity_threshold

    def _simple_overlap_detection(self, prev_text, curr_text):
        """
        Perform simple overlap detection as a fallback when LLM fails.

        Args:
            prev_text (str): Previous transcription text
            curr_text (str): Current transcription text

        Returns:
            str: The unique content from current transcription
        """
        # If current is shorter than previous, probably can't extract meaningful overlap
        if len(curr_text) <= len(prev_text) / 2:
            return curr_text

        # Split into words for comparison
        prev_words = prev_text.split()
        curr_words = curr_text.split()

        # Try to find longest common prefix
        common_length = 0
        for i in range(min(len(prev_words), len(curr_words))):
            if prev_words[-i - 1 :] == curr_words[: i + 1]:
                common_length = i + 1

        # If we found a substantial overlap, return only the unique part
        if common_length > 1 and common_length < len(curr_words):
            unique_part = " ".join(curr_words[common_length:])
            debug_utils.emit_debug(
                f"Simple overlap detection found {common_length} common words, {len(curr_words) - common_length} unique words",
                "info",
            )
            return unique_part

        # If no substantial overlap found, return the current text
        return curr_text

    def get_transcript_with_metadata(self, use_timestamps=False):
        """
        Get a transcript with timestamps and metadata.

        Args:
            use_timestamps (bool): Whether to include timestamps in the output

        Returns:
            List or str: List of transcriptions with timestamps, or combined text
        """
        with self.lock:
            if use_timestamps:
                # Track processed segments by ID
                processed_ids = set()

                # First pass: collect all valid segments
                valid_segments = []
                for item in self.full_history:
                    # Skip intermediate results
                    if item.get("is_intermediate"):
                        continue

                    # Skip segments that were merged
                    if item.get("was_merged"):
                        continue

                    # Get the text and clean it
                    text = item.get("text", "").strip()
                    if not text:
                        continue

                    # Skip segments containing LLM analysis text
                    if "The unique new content in the CURRENT TRANSCRIPTION" in text:
                        continue

                    if "<<NO_NEW_CONTENT>>" in text:
                        continue

                    # Use the segment_id to avoid duplicates
                    segment_id = item.get("segment_id")
                    if segment_id and segment_id in processed_ids:
                        continue

                    if segment_id:
                        processed_ids.add(segment_id)

                    # Add to valid segments with exact timestamps
                    valid_segments.append(
                        {
                            "text": text,
                            "start_time": float(item.get("start_time", 0)),
                            "end_time": float(item.get("end_time", 0)),
                            "is_merged": item.get("is_merged", False),
                        }
                    )

                # Sort segments by start time
                sorted_segments = sorted(valid_segments, key=lambda x: x["start_time"])

                return sorted_segments
            else:
                # For plain text, construct clean text without LLM artifacts
                clean_segments = []
                for item in self.full_history:
                    text = item.get("text", "").strip()
                    # Skip segments with LLM analysis text
                    if (
                        not text
                        or "The unique new content in the CURRENT TRANSCRIPTION" in text
                        or "<<NO_NEW_CONTENT>>" in text
                    ):
                        continue
                    clean_segments.append(text)

                return " ".join(clean_segments)

    def _clean_text(self, text):
        """
        Clean up text to fix punctuation and formatting issues.

        Args:
            text (str): The text to clean

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Fix double periods
        text = text.replace("..", ".")

        # Fix periods followed by lowercase letters (except for common abbreviations)
        common_abbr = ["mr.", "mrs.", "dr.", "ms.", "st.", "ave.", "e.g.", "i.e."]
        for abbr in common_abbr:
            # Temporarily replace abbreviations to avoid processing them
            text = text.replace(abbr, f"__ABBR_{abbr.upper()}_ABBR__")

        # Fix periods followed by lowercase letters (they should be spaces)
        text = re.sub(r"\.([a-z])", r". \1", text)

        # Restore abbreviations
        for abbr in common_abbr:
            text = text.replace(f"__ABBR_{abbr.upper()}_ABBR__", abbr)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?;:])\s+", r"\1 ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text


class TranscriptionProcessor:
    """Class to handle audio transcription using OpenAI's Whisper API."""

    def __init__(self, api_key=None, temp_dir=None):
        """Initialize the transcription processor.

        Args:
            api_key (str, optional): OpenAI API key. If None, uses OPENAI_API_KEY env var.
            temp_dir (str, optional): Directory to store temporary files.
                                     If None, system temp directory is used.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and not found in environment variables"
            )

        # Set up OpenAI API client
        openai.api_key = self.api_key

        # Initialize whisper model
        self.model_name = "whisper-1"
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.memory = TranscriptionMemory()
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.temp_files = []

        # Queues for audio chunks
        self.transcription_queue = Queue()
        self.multi_format_queue = Queue()

        # Thread synchronization
        self.lock = threading.RLock()  # Use RLock for better reentrant behavior

        # Add tracking for processed chunks to avoid duplicates
        self.processed_chunks = set()
        self.max_processed_chunks = 100  # Maximum number of processed chunks to track

        # Detailed debug logging
        self.debug_word_confidence = {}  # Track confidence scores for each word
        self.debug_chunk_boundaries = []  # Track chunk boundaries

        debug_utils.emit_debug(
            f"TranscriptionProcessor initialized with temp directory: {self.temp_dir}",
            "info",
        )

        # Test the API connection
        self.test_api()

    def test_api(self):
        """Test the OpenAI API connection with a simple request."""
        try:
            debug_utils.emit_debug("Testing OpenAI API connection...", "api")

            # Create a simple test audio file
            test_file_path = os.path.join(self.temp_dir, "test_audio.wav")

            # Generate a simple sine wave audio file
            # Generate a 1-second sine wave at 440 Hz
            sample_rate = 16000
            duration = 1  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = np.sin(2 * np.pi * 440 * t) * 32767
            audio_data = audio_data.astype(np.int16)

            # Save as WAV file
            wavfile.write(test_file_path, sample_rate, audio_data)

            debug_utils.emit_debug(
                f"Created test audio file: {test_file_path}",
                "api",
                {"file_size": os.path.getsize(test_file_path)},
            )

            # Test the API with the test file
            with open(test_file_path, "rb") as audio_file:
                debug_utils.emit_debug(
                    "Sending test request to OpenAI Whisper API", "api"
                )
                response = openai.Audio.transcribe(model=WHISPER_MODEL, file=audio_file)

            debug_utils.emit_debug(
                f"OpenAI API test successful: {response}",
                "api",
                {"response": str(response)},
            )

            # Clean up test file
            os.remove(test_file_path)
            debug_utils.emit_debug("Removed test audio file", "api")

            return True

        except Exception as e:
            debug_utils.emit_debug(
                f"OpenAI API test failed: {str(e)}",
                "error",
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            return False

    @debug_utils.debug_api_call
    def transcribe_audio_chunk(self, audio_data, start_time, end_time, metadata=None):
        """Transcribe an audio chunk using OpenAI's Whisper API.

        Args:
            audio_data (bytes): Raw audio data to transcribe
            start_time (float): Start time of the audio chunk in seconds
            end_time (float): End time of the audio chunk in seconds
            metadata (dict, optional): Additional metadata for the chunk

        Returns:
            dict: Transcription result with text and metadata
        """
        if metadata is None:
            metadata = {"pass": 1}

        pass_num = metadata.get("pass", 1)

        debug_utils.emit_debug(
            f"Starting transcription of audio chunk (pass {pass_num}): {start_time:.2f}s to {end_time:.2f}s, size: {len(audio_data)} bytes",
            "info",
            {
                "start_time": start_time,
                "end_time": end_time,
                "size": len(audio_data),
                "pass": pass_num,
            },
        )

        # Create a temporary file for the audio chunk
        temp_file = self._save_audio_to_temp_file(audio_data)
        self.temp_files.append(temp_file)

        try:
            debug_utils.emit_debug(
                f"Transcribing audio chunk (pass {pass_num}): {start_time:.2f}s to {end_time:.2f}s",
                "api",
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "file_size": os.path.getsize(temp_file),
                    "model": WHISPER_MODEL,
                    "file_path": temp_file,
                    "pass": pass_num,
                },
            )

            # Transcribe with retry logic
            result = self._transcribe_with_retry(temp_file)

            # Process and clean up the result
            transcription = self._process_transcription_result(result)

            debug_utils.emit_debug(
                f"Transcription completed (pass {pass_num}): {len(transcription)} characters",
                "api",
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "text_length": len(transcription),
                    "text_preview": transcription[:50]
                    + ("..." if len(transcription) > 50 else ""),
                    "pass": pass_num,
                },
            )

            # Create result dict with metadata
            result = {
                "text": transcription,
                "start_time": start_time,
                "end_time": end_time,
                "pass": pass_num,
                "timestamp": time.time(),
            }

            # For first pass, process with TranscriptionMemory
            if pass_num == 1:
                # Use the memory to merge with context
                result = self.memory.merge_transcription(result)

            return result

        except Exception as e:
            debug_utils.emit_debug(
                f"Error transcribing audio chunk (pass {pass_num}): {str(e)}",
                "error",
                {
                    "error": str(e),
                    "start_time": start_time,
                    "end_time": end_time,
                    "file_path": temp_file,
                    "pass": pass_num,
                },
            )
            raise

        finally:
            # Clean up the temporary file
            self._remove_temp_file(temp_file)
            if temp_file in self.temp_files:
                self.temp_files.remove(temp_file)

    def start_processing(self, output_callback):
        """Start the transcription processing thread.

        Args:
            output_callback (callable): Callback function to receive transcription results
        """
        # Use threading lock for thread safety
        with threading.Lock():
            # Check if thread is already running
            if self.processing_thread and self.processing_thread.is_alive():
                debug_utils.emit_debug(
                    "Transcription processing thread already running", "warn"
                )
                return

            debug_utils.emit_debug(
                f"Starting transcription processing with callback: {output_callback.__name__ if hasattr(output_callback, '__name__') else 'anonymous'}",
                "info",
            )

            # Clear any existing stop events
            self.stop_event.clear()

            # Create and start the processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_thread, args=(output_callback,), daemon=True
            )
            self.processing_thread.start()

            # Verify thread started successfully
            if self.processing_thread.is_alive():
                debug_utils.emit_debug(
                    "Started transcription processing thread", "info"
                )
            else:
                debug_utils.emit_debug(
                    "Failed to start transcription processing thread", "error"
                )

    def stop_processing(self):
        """Stop the transcription processing thread."""
        with self.lock:  # Use instance lock for consistent locking
            if self.processing_thread and self.processing_thread.is_alive():
                debug_utils.emit_debug(
                    "Stopping transcription processing thread", "info"
                )

                # Signal the thread to stop
                self.stop_event.set()

                # Wait for thread to finish with timeout
                start_time = time.time()
                max_wait = 10  # Maximum seconds to wait

                try:
                    self.processing_thread.join(timeout=max_wait)

                    # Check if thread is still alive after join attempt
                    if self.processing_thread.is_alive():
                        # Thread is stuck, we'll log but continue shutdown
                        debug_utils.emit_debug(
                            "Transcription thread did not terminate gracefully within timeout",
                            "warning",
                        )
                    else:
                        debug_utils.emit_debug(
                            f"Transcription processing thread stopped after {time.time() - start_time:.2f}s",
                            "info",
                        )
                except Exception as e:
                    debug_utils.emit_debug(
                        f"Error while stopping transcription thread: {str(e)}",
                        "error",
                        {"error": str(e), "traceback": traceback.format_exc()},
                    )

            # Clean up resources regardless of whether the thread stopped gracefully
            self._cleanup_temp_files()

            # Clear the queues to free memory
            try:
                while not self.transcription_queue.empty():
                    self.transcription_queue.get_nowait()
                while not self.multi_format_queue.empty():
                    self.multi_format_queue.get_nowait()
            except Exception as e:
                debug_utils.emit_debug(f"Error clearing queues: {str(e)}", "warning")

    def add_audio_chunk(self, audio_data, start_time, end_time):
        """Add an audio chunk to the transcription queue.

        Args:
            audio_data (bytes): Raw audio data to transcribe
            start_time (float): Start time of the audio chunk in seconds
            end_time (float): End time of the audio chunk in seconds
        """
        # Create a unique identifier for this chunk
        chunk_id = f"{start_time:.3f}_{end_time:.3f}"
        chunk_duration = end_time - start_time

        # Check if we've already processed this chunk
        if chunk_id in self.processed_chunks:
            debug_utils.emit_debug(
                f"Skipping already processed chunk: {start_time:.2f}s to {end_time:.2f}s",
                "info",
            )
            return

        debug_utils.emit_debug(
            f"Adding audio chunk to queue: {start_time:.2f}s to {end_time:.2f}s, size: {len(audio_data)} bytes",
            "info",
            {
                "start_time": start_time,
                "end_time": end_time,
                "duration": chunk_duration,
                "size": len(audio_data),
                "queue_size": self.transcription_queue.qsize(),
            },
        )

        # Track this chunk boundary for debugging
        self.debug_chunk_boundaries.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "duration": chunk_duration,
                "timestamp": time.time(),
            }
        )

        # Add to queue with metadata
        self.transcription_queue.put(
            (audio_data, start_time, end_time, {"pass": 1, "chunk_id": chunk_id})
        )

        # Add to processed chunks set
        self.processed_chunks.add(chunk_id)

        # Limit the size of processed_chunks
        if len(self.processed_chunks) > self.max_processed_chunks:
            self.processed_chunks.pop()

    def add_multi_format_audio_chunk(
        self, audio_data, start_time, end_time, sample_rate=16000, channels=1
    ):
        """Add a multi-format audio chunk for second-pass transcription.

        Args:
            audio_data (bytes): Raw audio data to transcribe
            start_time (float): Start time of the audio chunk in seconds
            end_time (float): End time of the audio chunk in seconds
            sample_rate (int): Sample rate of the audio
            channels (int): Number of audio channels
        """
        # Create metadata for this format
        format_metadata = {
            "pass": 2,  # Second pass
            "sample_rate": sample_rate,
            "channels": channels,
            "chunk_id": f"{start_time:.3f}_{end_time:.3f}_fmt_{sample_rate}_{channels}",
        }

        debug_utils.emit_debug(
            f"Adding multi-format audio chunk to queue: {start_time:.2f}s to {end_time:.2f}s, size: {len(audio_data)} bytes",
            "info",
            {
                "start_time": start_time,
                "end_time": end_time,
                "size": len(audio_data),
                "sample_rate": sample_rate,
                "channels": channels,
                "queue_size": self.multi_format_queue.qsize(),
            },
        )

        # Add to multi-format queue
        self.multi_format_queue.put((audio_data, start_time, end_time, format_metadata))

    def _contains_new_content(self, new_text, context):
        """Check if new_text contains content not present in context.

        Args:
            new_text (str): The new transcription text
            context (str): The existing context to compare against

        Returns:
            bool: True if new content is found, False otherwise
        """
        if not new_text or not new_text.strip():
            return False

        # If no context, all content is new
        if not context or not context.strip():
            return True

        # Normalize and clean text
        def normalize_text(text):
            return " ".join(text.lower().split())

        normalized_new = normalize_text(new_text)
        normalized_context = normalize_text(context)

        # Simple approach: check if the new text is entirely contained in the context
        if normalized_new in normalized_context:
            debug_utils.emit_debug("New text is fully contained in context", "info")
            return False

        # Check for at least some unique words (4+ characters to avoid small words)
        new_words = set(w for w in normalized_new.split() if len(w) >= 4)
        context_words = set(w for w in normalized_context.split() if len(w) >= 4)

        unique_words = new_words - context_words

        # If we have enough unique substantial words, consider it new content
        has_unique_content = len(unique_words) >= 2

        if has_unique_content:
            debug_utils.emit_debug(
                f"Found {len(unique_words)} unique substantial words in new content",
                "info",
                {"sample_words": list(unique_words)[:3] if unique_words else []},
            )
        else:
            debug_utils.emit_debug(
                "No significant new content found in segment", "info"
            )

        return has_unique_content

    def _processing_thread(self, output_callback):
        """Thread function for processing audio chunks in the queue.

        Args:
            output_callback (callable): Callback function to receive transcription results
        """
        debug_utils.emit_debug("Transcription processing thread started", "info")

        while not self.stop_event.is_set():
            try:
                # Process primary queue first (first pass)
                try:
                    debug_utils.emit_debug(
                        f"Checking primary queue (size: {self.transcription_queue.qsize()})",
                        "info",
                    )
                    audio_data, start_time, end_time, metadata = (
                        self.transcription_queue.get(timeout=0.5)
                    )
                    debug_utils.emit_debug(
                        f"Got audio chunk from primary queue: {start_time:.2f}s to {end_time:.2f}s, pass: {metadata.get('pass', 1)}",
                        "info",
                    )

                    # Process the audio chunk
                    self._process_single_chunk(
                        audio_data, start_time, end_time, metadata, output_callback
                    )
                    continue  # Continue to next iteration to prioritize first pass processing
                except Empty:
                    pass  # No items in primary queue, check secondary queue

                # Process secondary queue (multi-format queue)
                try:
                    debug_utils.emit_debug(
                        f"Checking multi-format queue (size: {self.multi_format_queue.qsize()})",
                        "info",
                    )
                    audio_data, start_time, end_time, metadata = (
                        self.multi_format_queue.get(timeout=0.5)
                    )
                    debug_utils.emit_debug(
                        f"Got audio chunk from multi-format queue: {start_time:.2f}s to {end_time:.2f}s, pass: {metadata.get('pass', 2)}",
                        "info",
                    )

                    # Process the audio chunk
                    self._process_single_chunk(
                        audio_data, start_time, end_time, metadata, output_callback
                    )
                except Empty:
                    # No items in either queue, sleep briefly
                    time.sleep(0.1)

            except Exception as e:
                debug_utils.emit_debug(
                    f"Error in transcription processing thread: {str(e)}",
                    "error",
                    {"error": str(e), "traceback": traceback.format_exc()},
                )
                # Sleep briefly to avoid tight loop in case of repeated errors
                time.sleep(0.5)

        debug_utils.emit_debug("Transcription processing thread stopped", "info")

    def _process_single_chunk(
        self, audio_data, start_time, end_time, metadata, output_callback
    ):
        """Process a single audio chunk from either queue.

        Args:
            audio_data (bytes): Audio data to transcribe
            start_time (float): Start time of the chunk
            end_time (float): End time of the chunk
            metadata (dict): Metadata for the chunk
            output_callback (callable): Callback for results
        """
        try:
            # Transcribe the audio chunk
            pass_num = metadata.get("pass", 1)
            debug_utils.emit_debug(
                f"Processing audio chunk (pass {pass_num}): {start_time:.2f}s to {end_time:.2f}s",
                "info",
            )

            result = self.transcribe_audio_chunk(
                audio_data, start_time, end_time, metadata
            )

            debug_utils.emit_debug(
                f"Successfully transcribed audio chunk (pass {pass_num}): {start_time:.2f}s to {end_time:.2f}s",
                "info",
            )

            # Call the output callback with the result
            if output_callback and result:
                text = result.get("text", "")

                # For pass 1 (primary), we use the memory to handle deduplication
                if pass_num == 1:
                    # Only call the callback if there's text content
                    if text and text.strip():
                        debug_utils.emit_debug(
                            f"Calling output callback with transcription result (pass {pass_num})",
                            "info",
                            {
                                "start_time": result.get("start_time"),
                                "end_time": result.get("end_time"),
                                "text_length": len(text),
                                "is_merged": result.get("is_merged", False),
                            },
                        )
                        output_callback(result)
                # For pass 2 (multi-format), only call if significant differences
                else:
                    # Get the context to compare with
                    context = self.memory.get_context(
                        50
                    )  # Last 50 words for comparison

                    # If the result contains words not in context, add it
                    if self._contains_new_content(text, context):
                        debug_utils.emit_debug(
                            f"Multi-format pass found new content (pass {pass_num})",
                            "info",
                            {"text": text[:50] + ("..." if len(text) > 50 else "")},
                        )

                        # Mark this as a secondary pass result
                        result["is_secondary_pass"] = True
                        output_callback(result)
                    else:
                        debug_utils.emit_debug(
                            f"Multi-format pass did not find new content (pass {pass_num})",
                            "info",
                        )

        except Exception as e:
            debug_utils.emit_debug(
                f"Error processing audio chunk: {str(e)}",
                "error",
                {"error": str(e), "traceback": traceback.format_exc()},
            )

    def _save_audio_to_temp_file(self, audio_data):
        """Save audio data to a temporary file.

        Args:
            audio_data (bytes): Raw audio data

        Returns:
            str: Path to the temporary file
        """
        # Create a temporary file with .wav extension
        fd, temp_path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
        os.close(fd)

        # Write the audio data to the file
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        file_size = os.path.getsize(temp_path)
        debug_utils.emit_debug(
            f"Saved audio chunk to temporary file: {temp_path}, size: {file_size} bytes",
            "info",
            {"file_path": temp_path, "file_size": file_size},
        )

        return temp_path

    def _remove_temp_file(self, file_path):
        """Remove a temporary file.

        Args:
            file_path (str): Path to the temporary file
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                debug_utils.emit_debug(f"Removed temporary file: {file_path}", "info")
            else:
                debug_utils.emit_debug(
                    f"Temporary file not found for removal: {file_path}", "warn"
                )
        except Exception as e:
            debug_utils.emit_debug(
                f"Error removing temporary file {file_path}: {str(e)}",
                "error",
                {"error": str(e), "file_path": file_path},
            )

    def _cleanup_temp_files(self):
        """Clean up all temporary files."""
        debug_utils.emit_debug(
            f"Cleaning up {len(self.temp_files)} temporary files", "info"
        )
        for file_path in self.temp_files[:]:
            self._remove_temp_file(file_path)
            self.temp_files.remove(file_path)

    @debug_utils.debug_api_call
    def _transcribe_with_retry(self, audio_file_path):
        """Transcribe an audio file with retry logic.

        Args:
            audio_file_path (str): Path to the audio file

        Returns:
            dict: Transcription result from the API

        Raises:
            Exception: If transcription fails after all retries
        """
        file_size = os.path.getsize(audio_file_path)
        debug_utils.emit_debug(
            f"Starting transcription with Whisper API for file: {audio_file_path}, size: {file_size} bytes",
            "api",
            {
                "file_path": audio_file_path,
                "file_size": file_size,
                "model": WHISPER_MODEL,
            },
        )

        for attempt in range(MAX_RETRIES):
            try:
                debug_utils.emit_debug(
                    f"Transcription attempt {attempt+1}/{MAX_RETRIES}", "api"
                )

                with open(audio_file_path, "rb") as audio_file:
                    # Call the OpenAI API to transcribe the audio (v0.27.0 syntax)
                    debug_utils.emit_debug(
                        f"Sending request to OpenAI Whisper API", "api"
                    )
                    response = openai.Audio.transcribe(
                        model=WHISPER_MODEL, file=audio_file
                    )

                debug_utils.emit_debug(
                    f"Transcription successful on attempt {attempt+1}",
                    "api",
                    {"response_type": type(response).__name__},
                )

                return response

            except openai.error.RateLimitError as e:
                wait_time = RETRY_DELAY * (2**attempt)  # Exponential backoff
                debug_utils.emit_debug(
                    f"Rate limit exceeded, retrying in {wait_time}s (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}",
                    "warn",
                    {"error": str(e), "wait_time": wait_time},
                )
                time.sleep(wait_time)

            except Exception as e:
                debug_utils.emit_debug(
                    f"Error transcribing audio (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}",
                    "error",
                    {"error": str(e), "error_type": type(e).__name__},
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise

        error_msg = f"Failed to transcribe audio after {MAX_RETRIES} attempts"
        debug_utils.emit_debug(error_msg, "error")
        raise Exception(error_msg)

    def _process_transcription_result(self, result):
        """Process and clean up the transcription result.

        Args:
            result (dict): Raw transcription result from the API

        Returns:
            str: Cleaned transcription text
        """
        debug_utils.emit_debug(
            f"Processing transcription result: {type(result).__name__}",
            "info",
            {
                "result_type": type(result).__name__,
                "result_preview": (
                    str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                ),
            },
        )

        # Extract the text from the result
        if isinstance(result, dict) and "text" in result:
            text = result["text"]
            debug_utils.emit_debug(f"Extracted text from result dictionary", "info")
        else:
            text = str(result)
            debug_utils.emit_debug(
                f"Converted result to string: {text[:50]}...", "warn"
            )

        # Clean up the text (remove extra whitespace, etc.)
        text = text.strip()

        debug_utils.emit_debug(
            f"Processed transcription result: {len(text)} characters",
            "info",
            {
                "text_length": len(text),
                "text_preview": text[:50] + ("..." if len(text) > 50 else ""),
            },
        )

        return text
