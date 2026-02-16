import cv2
import threading
import queue
import time
import json
from src.facial_emotion import FacialEmotion
from src.pose_gesture import PoseGestureEstimator
from src.audio_emotion import AudioEmotionProcessor
from src.utils import ts_now, write_json_line, LatestAudio

OUTPUT_PATH = "output.json"

def video_worker(latest_audio_store: LatestAudio, stop_event: threading.Event):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[main] Cannot open camera")
        return
    # Initialize processors
    fe = FacialEmotion(model_path="models/model_best.keras")  # optional path
    pg = PoseGestureEstimator()
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[main] failed to read frame")
                break
            # OpenCV gives BGR; convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = ts_now()
            face_res = fe.process(frame_rgb)
            landmarks = pg.process(frame_rgb)
            audio_emotion = latest_audio_store.get()
            out = {
                "timestamp": timestamp,
                "facial_emotion": face_res,
                "audio_emotion": audio_emotion,
                "landmarks": landmarks,
            }
            line = write_json_line(OUTPUT_PATH, out)
            print(line)
            # Small sleep to reduce CPU usage; this controls frame processing cadence
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("[main] video interrupted")
    finally:
        fe.close()
        pg.close()
        cap.release()

def audio_segment_callback(wav_path, sr, latest_audio_store: LatestAudio):
    # invoked when a speech segment is detected and saved to a WAV file
    ap = audio_proc
    res = ap.analyze_file(wav_path)
    # attach timestamp
    res_with_ts = {"timestamp": ts_now(), "result": res}
    latest_audio_store.set(res_with_ts)
    # Also write a JSON line specifically for audio (optional)
    out = {
        "timestamp": res_with_ts["timestamp"],
        "facial_emotion": None,
        "audio_emotion": res,
        "landmarks": None,
    }
    line = write_json_line(OUTPUT_PATH, out)
    print(line)

if __name__ == "__main__":
    stop_event = threading.Event()
    latest_audio = LatestAudio()
    # Create audio processor
    audio_proc = AudioEmotionProcessor(sr=16000)
    # Start audio stream in separate thread
    def audio_thread_fn():
        try:
            audio_proc.start_stream(lambda wav_path, sr: audio_segment_callback(wav_path, sr, latest_audio))
            # Keep running until stop_event
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            audio_proc.stop()

    audio_thread = threading.Thread(target=audio_thread_fn, daemon=True)
    audio_thread.start()

    try:
        video_worker(latest_audio, stop_event)
    except KeyboardInterrupt:
        print("[main] Interrupted by user")
    finally:
        stop_event.set()
        audio_proc.stop()
        time.sleep(0.5)
        print("[main] exiting")
