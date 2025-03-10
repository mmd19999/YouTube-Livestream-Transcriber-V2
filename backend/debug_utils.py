#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Utilities for YouTube Livestream Transcriber

This module provides debug utilities for the YouTube Livestream Transcriber application,
including functions for logging and emitting debug messages to clients.
"""

import logging
import time
import traceback
import json
from datetime import datetime
from functools import wraps

# Initialize logger
logger = logging.getLogger("debug")

# Global reference to socketio instance (set by app.py)
socketio = None


def set_socketio(socketio_instance):
    """Set the global socketio instance

    Args:
        socketio_instance: The SocketIO instance to use for emitting messages
    """
    global socketio
    socketio = socketio_instance


def emit_debug(message, type="info", data=None):
    """Emit a debug message to all connected clients

    Args:
        message (str): The debug message
        type (str, optional): Message type (info, warn, error, api, llm). Defaults to "info".
        data (dict, optional): Additional data to include. Defaults to None.
    """
    # Log to server console
    log_method = getattr(logger, type if type in ["info", "warn", "error"] else "info")
    log_method(f"[DEBUG] {message}")

    # Emit to clients if socketio is available
    if socketio:
        socketio.emit(
            "debug_message",
            {
                "message": message,
                "type": type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            },
        )


def debug_api_call(func):
    """Decorator to log API calls with timing information

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function name for logging
        func_name = func.__name__

        # Format args and kwargs for better readability
        args_str = str(args)
        if len(args_str) > 500:
            args_str = args_str[:500] + "..."

        kwargs_str = json.dumps(kwargs, default=str)
        if len(kwargs_str) > 500:
            kwargs_str = kwargs_str[:500] + "..."

        # Log start of API call
        emit_debug(
            f"API call started: {func_name}",
            "api",
            {"args": args_str, "kwargs": kwargs_str, "function": func.__qualname__},
        )

        # Start timing
        start_time = time.time()

        try:
            # Call the original function
            result = func(*args, **kwargs)

            # Calculate duration
            duration = time.time() - start_time

            # Format result for logging
            result_preview = str(result)
            if len(result_preview) > 500:
                result_preview = result_preview[:500] + "..."

            # Log successful API call
            emit_debug(
                f"API call completed: {func_name}",
                "api",
                {
                    "duration_ms": round(duration * 1000),
                    "result_type": type(result).__name__,
                    "result_preview": (
                        result_preview
                        if not isinstance(result, bytes)
                        else f"<binary data of length {len(result)}>"
                    ),
                    "duration_seconds": round(duration, 3),
                },
            )

            return result
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Log failed API call
            emit_debug(
                f"API call failed: {func_name}",
                "error",
                {
                    "duration_ms": round(duration * 1000),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
            raise

    return wrapper


def debug_llm_process(action):
    """Decorator to log LLM processing with timing information

    Args:
        action (str): The LLM action being performed

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log start of LLM process
            emit_debug(
                f"LLM process started: {action}",
                "llm",
                {"args": str(args), "kwargs": str(kwargs)},
            )

            # Start timing
            start_time = time.time()

            try:
                # Call the original function
                result = func(*args, **kwargs)

                # Calculate duration
                duration = time.time() - start_time

                # Log successful LLM process
                emit_debug(
                    f"LLM process completed: {action}",
                    "llm",
                    {
                        "duration_ms": round(duration * 1000),
                        "result_type": type(result).__name__,
                    },
                )

                return result

            except Exception as e:
                # Calculate duration
                duration = time.time() - start_time

                # Get traceback
                tb = traceback.format_exc()

                # Log error
                emit_debug(
                    f"LLM process failed: {action} - {str(e)}",
                    "error",
                    {
                        "duration_ms": round(duration * 1000),
                        "error": str(e),
                        "traceback": tb,
                    },
                )

                # Re-raise the exception
                raise

        return wrapper

    return decorator


def log_verification(status, details=None):
    """Log a verification process event

    Args:
        status (str): The verification status
        details (dict, optional): Additional details about the verification. Defaults to None.
    """
    emit_debug(
        f"Verification process: {status}",
        "info",
        details,
    )


def format_json_for_debug(obj):
    """Format an object as JSON for debug output

    Args:
        obj: The object to format

    Returns:
        str: The formatted JSON string
    """
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception as e:
        return f"Error formatting JSON: {str(e)}"
