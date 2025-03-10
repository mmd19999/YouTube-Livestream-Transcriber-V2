# YouTube Livestream Transcriber - Project Goals

## Project Overview

We're building a real-time YouTube livestream transcription application that connects to YouTube livestreams, extracts the audio, and provides accurate transcription using OpenAI's Whisper API. The application will also automatically detect topic changes in the conversation and organize content accordingly.

## Current Progress (Updated)

- ✅ **YouTube Livestream Connection**: Successfully implemented connection to YouTube livestreams using yt-dlp and FFmpeg for audio extraction
- ✅ **Audio Processing Pipeline**: Implemented segmentation and conversion of audio chunks with optimized overlapping
- ✅ **Real-time Transcription**: Integrated OpenAI's Whisper API for audio transcription
- ✅ **Socket.IO Communication**: Set up real-time updates between server and client
- ✅ **Deduplication System**: Added intelligent filtering to prevent repeated transcriptions
- ✅ **Chunk Overlap Optimization**: Implemented 15-second chunks with 10-second overlaps (5-second advancement)
- ⚠️ **Transcription Accuracy**: Significantly improved through frequent chunk processing, but still refining
- ⏳ **Topic Change Detection**: Not yet implemented
- ⏳ **Advanced UI Features**: Basic UI implemented, advanced features pending

## Current Sprint Goals

- ✅ **Real-time Transcription**: Integrated OpenAI's Whisper API for audio transcription
- ✅ **YouTube Integration**: Connected to YouTube livestreams for audio extraction
- ✅ **UI Development**: Created responsive interface for displaying transcriptions
- ✅ **Overlap Optimization**: Implemented LLM-based overlap resolution to prevent content loss
- ✅ **Chunk Processing Optimization**: Implemented 5-second advancement between 15-second chunks
- ⬜ **Topic Detection**: Detect topic changes in the conversation
- ⬜ **Content Organization**: Group transcribed content by topics

## Key Features to Implement

1. **YouTube Livestream Connection**
   - ✅ Connect to YouTube livestreams using yt-dlp and cookie authentication
   - ✅ Extract audio for processing using FFmpeg

2. **Real-time Transcription**
   - ✅ Transcribe speech in real-time using OpenAI's Whisper API
   - ✅ Process audio in chunks
   - ✅ Optimize chunk overlapping to prevent missed content
   - ⚠️ Continue refining accuracy for missed words and sentence endings

3. **Intelligent Memory System**
   - ✅ Implement deduplication to eliminate repetitions
   - ✅ Implement LLM-based overlap resolution
   - ✅ Optimize chunk processing frequency (5-second advancement)
   - ⏳ Enhance merging of transcription segments

4. **Topic Change Detection**
   - ⏳ Automatically identify conversation topic changes
   - ⏳ Use dual-LLM verification for accuracy (mark verified topics with 🤖🤖)
   - ⏳ Generate descriptive topic titles
   - ⏳ Display topics with timestamps and detection reasons

5. **User Interface**
   - ✅ Create basic interface with transcription display
   - ⏳ Implement dark mode support
   - ⏳ Add interactive topic history with detection information
   - ✅ Include debug console for troubleshooting
   - ⏳ Implement copy/save functionality for transcriptions

## Technical Architecture

- **Backend**: Flask + Socket.IO server with Python
- **Frontend**: HTML, CSS, JavaScript with Socket.IO client
- **AI Integration**: OpenAI Whisper API and GPT-3.5-Turbo for overlap resolution
- **Media Processing**: FFmpeg for audio extraction, yt-dlp for stream access
- **Audio Processing**: 15-second chunks with 10-second overlaps (5-second advancement)

## Next Steps

1. **Further Improve Transcription Accuracy**
   - ✅ Implemented optimized chunk overlapping (15s chunks, 10s overlap, 5s advancement)
   - Continue refining audio processing parameters
   - Implement post-processing to correct common errors
   - Consider context-aware merging of sentence fragments

2. **Implement Topic Detection**
   - Set up the dual-LLM verification system
   - Develop algorithms for identifying topic changes
   - Create UI components for displaying topic history

3. **Enhance User Interface**
   - Add dark mode support
   - Implement copy/save functionality
   - Improve responsive design for different screen sizes

4. **Performance Optimization**
   - Reduce latency in transcription pipeline
   - Optimize memory usage for long sessions
   - Improve error handling and recovery

This project aims to make livestream content more accessible and organized by providing real-time transcription with intelligent topic organization.

## Technical Tasks

- ✅ Set up Flask + Socket.IO backend
- ✅ Implement YouTube stream extraction with yt-dlp
- ✅ Transcribe speech in real-time using OpenAI's Whisper API
- ✅ Create responsive UI with real-time updates
- ✅ Optimize chunk overlapping to prevent missed content
- ⬜ Implement topic detection using LLM
- ⬜ Add topic visualization in the UI

## Future Enhancements

- ⏳ Implement multi-language support
- ⏳ Add export functionality (TXT, SRT, etc.)
- ⏳ Develop speaker identification
- ⏳ Create user accounts for saved transcripts
- ⏳ Implement advanced LLM prompt optimization for overlap detection