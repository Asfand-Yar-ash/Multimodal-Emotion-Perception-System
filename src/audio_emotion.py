import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import sys
import threading
import time
import tempfile
import soundfile as sf
from transformers import pipeline

# Change this to a preferred HF SER model for audio-classification
HF_SER_MODEL = "superb/hubert-large-superb-er"  # example; change if required

class AudioEmotionProcessor:
    def __init__(self, sr=16000, device=None):
        self.sr = sr
        self.device = device
        self.vad = webrtcvad.Vad(3)  # aggressiveness 0-3
        self.frame_duration_ms = 30
        self.frame_bytes = int(self.sr * (self.frame_duration_ms / 1000.0) * 2)  # 16-bit
        self.buffer_lock = threading.Lock()
        self.recording = False
        self._stop = threading.Event()
        # Initialize HF pipeline (lazy init)
        try:
            self.pipeline = pipeline("audio-classification", model=HF_SER_MODEL, top_k=3)
            print(f"[audio_emotion] Loaded HF audio-classification pipeline with model {HF_SER_MODEL}")
        except Exception as e:
            print(f"[audio_emotion] Failed to create HF pipeline: {e}")
            self.pipeline = None

    def _bytes_to_pcm16(self, audio_frame):
        # audio_frame: float32 numpy array, shape (frames, channels)
        if audio_frame.ndim > 1:
            audio_frame = np.mean(audio_frame, axis=1)
        pcm16 = (audio_frame * 32767).astype(np.int16)
        return pcm16.tobytes()
    def start_stream(self, on_segment_callback):
        self.on_segment_callback = on_segment_callback
        self._stop.clear()

        self.in_speech = False
        self.speech_buffer = bytearray()
        self.vad_buffer = bytearray()

        FRAME_SAMPLES = int(self.sr * self.frame_duration_ms / 1000)
        FRAME_BYTES = FRAME_SAMPLES * 2
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[audio] status: {status}", file=sys.stderr)

            pcm_bytes = self._bytes_to_pcm16(indata)

            self.vad_buffer.extend(pcm_bytes)

            while len(self.vad_buffer) >= FRAME_BYTES:
                frame = self.vad_buffer[:FRAME_BYTES]
                self.vad_buffer = self.vad_buffer[FRAME_BYTES:]

                try:
                    is_speech = self.vad.is_speech(frame, self.sr)
                except Exception as e:
                    print("[audio] VAD error:", e)
                    self.in_speech = False
                    self.speech_buffer.clear()
                    continue

                if is_speech:
                    self.in_speech = True
                    self.speech_buffer.extend(frame)
                else:
                    if self.in_speech:
                        pcm16 = np.frombuffer(self.speech_buffer, dtype=np.int16).astype(np.float32) / 32767.0

                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
                                sf.write(tf.name, pcm16, self.sr, subtype="PCM_16")
                                if self.on_segment_callback:
                                    self.on_segment_callback(tf.name, self.sr)
                        except Exception as e:
                            print("[audio] wav write error:", e)

                        self.speech_buffer.clear()
                        self.in_speech = False

        self.stream = sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=self.sr,
            device=self.device,
            dtype="float32",
            blocksize=FRAME_SAMPLES
        )

        self.stream.start()
        print("[audio] stream started")

    def stop(self):
        try:
            if hasattr(self, "stream"):
                self.stream.stop()
                self.stream.close()
                print("[audio] stream stopped")
        except Exception as e:
            print("[audio] error stopping stream:", e)

    def analyze_file(self, wav_path):
        """
        Run HF pipeline on wav_path and return a dict with predictions and timestamp.
        """
        if self.pipeline is None:
            return {"label": None, "confidence": None, "details": None}
        try:
            results = self.pipeline(wav_path)
            # results is list of dicts {label, score}
            if results and isinstance(results, list):
                top = results[0]
                return {"label": top.get("label"), "confidence": float(top.get("score")), "details": results}
            else:
                return {"label": None, "confidence": None, "details": results}
        except Exception as e:
            print("[audio] pipeline error:", e)
            return {"label": None, "confidence": None, "details": str(e)}
