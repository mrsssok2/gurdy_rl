"""Smoke test for the Gnani Speech APIs (STT Prisma v3.0 + TTS Timbre v2.5).

Usage (Windows, inside the venv):
    $env:GNANI_API_KEY = "<your key>"
    pip install -r voice_client/requirements.txt
    python voice_client/test_gnani.py [--seconds 5] [--voice Nalini] [--language hi-IN]

The API key is read only from the GNANI_API_KEY environment variable and is
never written to disk. See docs/gnani/notes.md for the API details.
"""

import argparse
import io
import os
import sys
import tempfile
import wave

import numpy as np
import requests
import sounddevice as sd

BASE_URL = "https://api.vachana.ai"
STT_URL = f"{BASE_URL}/stt/v3"
TTS_URL = f"{BASE_URL}/api/v1/tts/inference"

REC_SAMPLE_RATE = 16000  # STT converts to 16 kHz mono anyway; send that directly
TTS_SAMPLE_RATE = 24000
TTS_TEXT = "Main ek AI assistant hoon. Main bees centimeter aage chal gaya."


def get_api_key():
    key = os.environ.get("GNANI_API_KEY", "").strip()
    if not key:
        sys.exit('GNANI_API_KEY is not set. In PowerShell: $env:GNANI_API_KEY = "<key>"')
    return key


def report_failure(name, resp):
    print(f"[{name}] FAILED: HTTP {resp.status_code}")
    print(f"[{name}] error body: {resp.text[:2000]}")


def record_wav(seconds):
    """Record mono 16-bit PCM from the default mic and return WAV bytes."""
    print(f"Recording {seconds} s from the default microphone... speak now.")
    audio = sd.rec(int(seconds * REC_SAMPLE_RATE), samplerate=REC_SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    print("Recording done.")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(REC_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def stt(api_key, wav_bytes):
    resp = requests.post(
        STT_URL,
        headers={"X-API-Key-ID": api_key},
        files={"audio_file": ("recording.wav", wav_bytes, "audio/wav")},
        data={"model": "prisma-v3.0", "language_code": "indic-auto"},
        timeout=60,
    )
    print(f"[STT] HTTP {resp.status_code}")
    if resp.status_code != 200:
        report_failure("STT", resp)
        return False
    body = resp.json()
    print(f"[STT] transcript:         {body.get('transcript')!r}")
    print(f"[STT] detected language:  {body.get('detected_language_code', '(none returned)')}")
    print(f"[STT] model: {body.get('model')}  request_id: {body.get('request_id')}  "
          f"latency: {body.get('end_to_end_latency')} s")
    return True


def tts(api_key, text, voice, language):
    payload = {
        "text": text,
        "voice": voice,
        "model": "timbre-v2.5",
        "language": language,
        "speed": 1.0,
        "audio_config": {
            "sample_rate": TTS_SAMPLE_RATE,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }
    resp = requests.post(
        TTS_URL,
        headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    ctype = resp.headers.get("Content-Type", "")
    print(f"[TTS] HTTP {resp.status_code}  Content-Type: {ctype}  bytes: {len(resp.content)}")
    if resp.status_code != 200 or "json" in ctype:
        report_failure("TTS", resp)
        return None
    return resp.content


def play_wav(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        sys.exit(f"Unexpected sample width {width} bytes in TTS audio")
    audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    print(f"Playing TTS audio ({len(audio) / rate:.2f} s, {rate} Hz, {channels} ch)...")
    sd.play(audio, rate)
    sd.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0, help="recording length (max 30)")
    ap.add_argument("--voice", default="Nalini", help="Timbre v2.5 voice (Hindi: Nalini, Deepak, ...)")
    ap.add_argument("--language", default="hi-IN", help="TTS language (hi-IN, hi-en, ...)")
    ap.add_argument("--skip-stt", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    args = ap.parse_args()
    if not 0.1 <= args.seconds <= 30:
        sys.exit("--seconds must be between 0.1 and 30 (STT REST limit)")

    api_key = get_api_key()
    ok = True

    if not args.skip_stt:
        wav_bytes = record_wav(args.seconds)
        path = os.path.join(tempfile.gettempdir(), "gnani_recording.wav")
        with open(path, "wb") as f:
            f.write(wav_bytes)
        print(f"Saved recording to {path}")
        ok &= stt(api_key, wav_bytes)

    if not args.skip_tts:
        print(f"[TTS] voice={args.voice} language={args.language} text={TTS_TEXT!r}")
        audio = tts(api_key, TTS_TEXT, args.voice, args.language)
        if audio is None:
            ok = False
        else:
            path = os.path.join(tempfile.gettempdir(), "gnani_tts.wav")
            with open(path, "wb") as f:
                f.write(audio)
            print(f"Saved TTS audio to {path}")
            play_wav(audio)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
