# Changelog

All notable changes to the YouTube Livestream Transcriber V2 project will be documented in this file.

## [Unreleased]

### Added
- Topic change detection system (planned)
- Export functionality for transcriptions (planned)
- Dark mode toggle (planned)

## [0.2.0] - 2025-03-10

### Added
- Exact timestamp preservation in transcriptions
- Frontend display of exact timestamps
- Enhanced debug console with message type filtering

### Changed
- Modified `get_transcript_with_metadata` function to preserve exact timestamps
- Updated frontend to display exact timestamps for each transcription segment
- Reduced advancement between chunks from 12 seconds to 5 seconds
- Reduced waiting time between chunk processing from 15 seconds to 5 seconds

### Fixed
- Fixed timestamp display to show exact timestamps rather than rounding to 15-second boundaries
- Improved handling of audio formats with null bitrate values

## [0.1.0] - 2025-03-09

### Added
- Initial implementation of YouTube livestream connection
- Audio extraction using FFmpeg
- Real-time transcription using OpenAI's Whisper API
- Socket.IO communication for real-time updates
- Basic deduplication system
- LLM-based overlap resolution
- Optimized audio chunking with 15-second segments and 10-second overlaps
- Debug console for troubleshooting

### Known Issues
- Some transcription accuracy issues with missed words or sentence endings
- FFmpeg extraction sometimes fails to capture audio cleanly
- Timestamps rounded to 15-second intervals 