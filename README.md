# ai-presentation-emotion-modules

Minimal starter implementation for capturing webcam + mic, running facial emotion (FER), pose & hand landmarks, and audio emotion recognition (SER), and emitting timestamped JSON lines with three module outputs:
- facial_emotion
- audio_emotion
- landmarks

This repository contains a beginner-friendly, extendable pipeline. It provides safe fallbacks when heavy model weights are not available, and shows where to plug in real models.

## Features (starter)
- Video capture (OpenCV) + MediaPipe Face Detection and Holistic (pose & hands)
- Facial emotion recognition (FER) placeholder that can load a Keras mini_XCEPTION model if provided
- Audio capture (sounddevice) + WebRTC VAD to segment speech + HuggingFace audio-classification pipeline for SER (configurable model)
- Parallel processing of audio and video
- Emits timestamped JSON Lines to `output.json` and prints each line to stdout

## Requirements

Python 3.8+

Install dependencies:
```
pip install -r requirements.txt
```

Note: GPU acceleration requires appropriate PyTorch + CUDA installation if you use GPU-enabled HuggingFace/Speech models.

## Models and configuration

- Facial emotion (FER / mini_XCEPTION):
  - If you have a trained mini_XCEPTION Keras model (typical FER models use 48x48 grayscale input and output 7 emotion classes), put it at `models/mini_xception.h5`. The code will attempt to load it. If not present, facial_emotion will return a neutral placeholder.

- Audio SER:
  - By default the code uses a HuggingFace model name configured in `src/audio_emotion.py`. You can change the `HF_SER_MODEL` constant to a different HF audio-classification model suitable for emotion recognition (e.g. `superb/hubert-large-superb-er` or any other).
  - If you prefer a SpeechBrain model, update `src/audio_emotion.py` accordingly.

## Running

From the repository root:
```
python src/main.py
```

This will:
- Start webcam capture (device 0 by default) and microphone capture.
- Print timestamped JSON lines to stdout.
- Append the same JSON lines to `output.json` (one JSON object per line).

Press Ctrl+C to stop.

## Files

- `src/main.py` - entrypoint
- `src/facial_emotion.py` - facial detection + FER model hook
- `src/pose_gesture.py` - MediaPipe Holistic (pose & hands)
- `src/audio_emotion.py` - audio capture + VAD + SER
- `src/utils.py` - utility functions
- `requirements.txt` - dependencies
- `.gitignore`, `LICENSE` - helpers

## Notes & Extensions

- The implementation is intentionally minimal: it aims to be readable and easy to adapt. Replace the placeholder FER model logic with a trained mini_XCEPTION weights file to get real facial emotion output.
- Consider model performance and CPU/GPU resource usage: heavy models will require GPU to run in real-time.
- For production or presentation use, add batching, result smoothing, timestamp alignment with higher precision, and a better mechanism for synchronizing audio segments with video frames.

## License

MIT License

Copyright (c) 2026 Asfand-Yar-ash

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
