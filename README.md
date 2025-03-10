# YouTube Livestream Transcriber V2

A real-time YouTube livestream transcription application that connects to YouTube livestreams, extracts the audio, and provides accurate transcription using OpenAI's Whisper API. The application preserves exact timestamps and implements intelligent overlap resolution for seamless transcription.

## 🚀 Current Status

### ✅ Working Features

- **YouTube Livestream Connection**: Successfully connects to YouTube livestreams using yt-dlp and cookie authentication
- **Audio Processing Pipeline**: Extracts and processes audio in optimized chunks with proper overlapping
- **Real-time Transcription**: Transcribes audio using OpenAI's Whisper API with exact timestamp preservation
- **Socket.IO Communication**: Provides real-time updates between server and client
- **Deduplication System**: Intelligently filters repeated content using similarity detection
- **LLM-based Overlap Resolution**: Uses GPT-3.5-Turbo to analyze overlapping segments and extract unique content
- **Optimized Chunk Processing**: Implements 15-second chunks with 10-second overlaps (5-second advancement)
- **Debug Console**: Provides comprehensive logging and debugging information

### ⚠️ Partially Working Features

- **Transcription Accuracy**: Significantly improved through frequent chunk processing, but occasional issues with missed words or sentence endings
- **Audio Format Handling**: Works with most YouTube formats but may encounter issues with some unusual formats

### ❌ Not Yet Implemented

- **Topic Change Detection**: Automatic identification of conversation topic changes
- **Advanced UI Features**: Copy/save functionality, dark mode toggle, responsive design improvements
- **Export Functionality**: Exporting transcripts to various formats (TXT, SRT, etc.)

## 🔧 Setup

1. Clone the repository:
   ```
   git clone https://github.com/mmd19999/YouTube-Livestream-Transcriber-V2.git
   cd YouTube-Livestream-Transcriber-V2
   ```

2. Install the required packages:
   ```
   cd backend
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key in the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. (Optional) Add YouTube cookies to `backend/youtube_cookies.txt` for authenticated access to restricted videos

## 🚀 Running the Application

1. Start the server:
   ```
   python backend/app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5050
   ```

3. Enter a YouTube livestream URL and click "Connect"

## 🧪 Testing

To test the basic Socket.IO connection:

1. Start the server as described above
2. Open your browser and navigate to http://localhost:5050
3. Check the connection status in the UI - it should show "Connected to server" in green
4. Open the browser console to see Socket.IO events
5. Try opening multiple browser tabs to see the connected clients count increase

## 📁 Project Structure

- `backend/` - Flask & Socket.IO server
  - `app.py` - Main server file
  - `audio_processor.py` - Audio extraction and processing from YouTube streams
  - `transcription.py` - Whisper API integration and transcription processing
  - `youtube_extractor.py` - YouTube URL validation and stream extraction
  - `debug_utils.py` - Debugging utilities and logging
  - `requirements.txt` - Required Python packages
  - `.env` - Environment variables (OpenAI API key)
- `frontend/` - HTML, CSS, and JavaScript files
  - `index.html` - Main HTML file
  - `static/` - Static assets (CSS, JS, images)

## 🔄 Audio Processing System

The application uses an optimized audio processing approach:
- 15-second audio chunks for consistent display
- 10-second overlap between consecutive chunks
- 5-second advancement between chunks
- LLM-powered overlap resolution to maintain narrative continuity
- Multiple audio format processing for improved transcription accuracy

## 🔍 Recent Updates

- **Exact Timestamp Preservation**: Modified the `get_transcript_with_metadata` function to preserve exact timestamps instead of rounding to 15-second intervals
- **Frontend Timestamp Display**: Updated the frontend to display exact timestamps for each transcription segment
- **LLM-based Overlap Resolution**: Implemented intelligent analysis of overlapping segments to extract unique content
- **Optimized Chunk Processing**: Reduced advancement between chunks from 12 seconds to 5 seconds for better continuity
- **Increased Processing Frequency**: Reduced waiting time between chunk processing from 15 seconds to 5 seconds

## 🐛 Known Issues

- Some transcription accuracy issues remain with occasional missed words or sentence endings
- FFmpeg extraction sometimes fails to capture audio cleanly
- Potential memory usage concerns for very long livestreams

## 🔮 Next Steps

- Implement topic change detection system
- Enhance the user interface with topic visualization
- Add export functionality for transcriptions
- Improve responsive design for different screen sizes
- Optimize memory usage for long sessions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the Whisper API
- yt-dlp for YouTube stream extraction
- FFmpeg for audio processing 