# MyTranscriber

**MyTranscriber** is a PyQt6-based transcription and translation tool that uses [OpenAI Whisper](https://github.com/openai/whisper). It supports:

- Local audio/video files
- Downloading online media with `yt-dlp`
- Minimal speaker diarization with `speechbrain`
- An embedded logo in the header

## Features

- Drag-and-Drop: Drop files onto the window or browse for them
- Basic Batch Mode: Process multiple files in one go
- Translation: Choose "Translate" for source-to-English
- Simple Configuration: YAML-based config plus PyQt `QSettings`
- macOS `.app`: Build once with py2app—no Terminal needed for end users

## Installation (for Development)

1. Clone the repo:
   ```bash
   git clone https://github.com/chunkymonkey-lab/MyTranscriber.git
   cd MyTranscriber
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your logo.png to the resources directory

4. Run:
   ```bash
   python multi_transcriber_enhanced.py
   ```

## Building for macOS

1. Install py2app:
   ```bash
   pip install py2app
   ```

2. Build:
   ```bash
   python setup.py py2app
   ```

3. Find MyTranscriber.app in the dist/ folder

## Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.