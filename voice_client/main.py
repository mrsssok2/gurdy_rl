"""Voice control for Gurdy: speak a Hindi/Hinglish command, the robot in the
ROS container walks, and the assistant answers in Hinglish.

Press Enter -> record 4 s -> Gnani STT (prisma-v3.0, hi-IN) -> parse
(command_parser) -> publish the transcript and command into the container
(docker exec ... rostopic pub) -> wait for /gurdy/voice_feedback -> build a
Hinglish reply (replies) -> Gnani TTS (timbre-v2.5, Poorvi, hi-en) -> play.

While Gurdy walks, press Enter again to speak a stop ("Gurdy ruko").

Usage (Windows, inside the venv, with voice_demo.launch running in the container):
    $env:GNANI_API_KEY = "<your key>"
    python voice_client/main.py
    python voice_client/main.py --type          # type transcripts, no mic / STT
    python voice_client/main.py --dry-run       # print docker commands, no robot

The API key is read only from GNANI_API_KEY and never written to disk.
"""
import argparse
import io
import os
import sys
import time
import uuid
import wave

import requests

import command_parser
import replies
from robot_link import DEFAULT_CONTAINER, DEFAULT_SETUP, RobotLink

try:
    import msvcrt  # Windows: listen for Enter while the robot walks
except ImportError:
    msvcrt = None

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
REC_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
RETRY_DELAYS = (1.0, 2.0, 4.0)  # after a 429 (one request at a time per account)

# Seconds to wait for the robot's reply. A walk of 50 cm is ~300 steps of
# 0.3 s sim time; leave room for a slow real-time factor.
WALK_TIMEOUT = 300.0
REPLY_TIMEOUT = 20.0


def banner(stage, text=""):
    print("\n[%s] %s" % (stage, text), flush=True)


def get_api_key():
    key = os.environ.get("GNANI_API_KEY", "").strip()
    if not key:
        sys.exit('GNANI_API_KEY is not set. In PowerShell: $env:GNANI_API_KEY = "<key>"')
    return key


def post_with_retry(name, **kwargs):
    """requests.post, retried after HTTP 429. Returns the last response."""
    for delay in RETRY_DELAYS + (None,):
        resp = requests.post(timeout=60, **kwargs)
        if resp.status_code != 429 or delay is None:
            return resp
        print("[%s] rate limited (429); retrying in %.0f s" % (name, delay), flush=True)
        time.sleep(delay)
    return resp


def record_wav(seconds):
    import sounddevice as sd
    banner("MIC", "Recording %.0f s... speak now." % seconds)
    audio = sd.rec(int(seconds * REC_SAMPLE_RATE), samplerate=REC_SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    print("[MIC] done", flush=True)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(REC_SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def stt(api_key, wav_bytes, language):
    """Transcript text, or None on failure."""
    resp = post_with_retry(
        "STT", url=STT_URL, headers={"X-API-Key-ID": api_key},
        files={"audio_file": ("command.wav", wav_bytes, "audio/wav")},
        data={"model": "prisma-v3.0", "language_code": language})
    if resp.status_code != 200:
        print("[STT] FAILED: HTTP %d %s" % (resp.status_code, resp.text[:500]), flush=True)
        return None
    return (resp.json().get("transcript") or "").strip()


def tts(api_key, text, voice, language):
    """WAV bytes, or None on failure."""
    resp = post_with_retry(
        "TTS", url=TTS_URL,
        headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"},
        json={"text": text, "voice": voice, "model": "timbre-v2.5", "language": language,
              "speed": 1.0,
              "audio_config": {"sample_rate": TTS_SAMPLE_RATE, "num_channels": 1,
                               "sample_width": 2, "encoding": "linear_pcm",
                               "container": "wav"}})
    if resp.status_code != 200 or "json" in resp.headers.get("Content-Type", ""):
        print("[TTS] FAILED: HTTP %d %s" % (resp.status_code, resp.text[:500]), flush=True)
        return None
    return resp.content


def play_wav(wav_bytes):
    import numpy as np
    import sounddevice as sd
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate, channels = wf.getframerate(), wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    sd.play(audio, rate)
    sd.wait()


def make_command(parsed, mode):
    """voice_command JSON dict for a parsed walk/stop/status, else None."""
    intent = parsed["intent"]
    cmd = {"id": uuid.uuid4().hex[:8]}
    if intent == "walk_forward":
        cmd.update(action="walk_forward", distance_m=parsed["distance_m"], mode=mode)
    elif intent in ("stop", "status"):
        cmd["action"] = intent
    else:
        return None
    return cmd


class VoiceDemo(object):
    def __init__(self, args):
        self.args = args
        self.api_key = None if (args.type and args.no_tts) else get_api_key()
        self.mode = args.mode
        self.robot = RobotLink(container=args.container, setup=args.setup,
                               echo_delay=args.echo_delay, dry_run=args.dry_run)

    def listen(self):
        """One transcript: typed (--type) or recorded and transcribed."""
        if self.args.type:
            return input("Type a command: ").strip()
        wav = record_wav(self.args.seconds)
        banner("STT", "prisma-v3.0, %s ..." % self.args.language)
        return stt(self.api_key, wav, self.args.language)

    def say(self, text):
        banner("REPLY", text)
        if self.args.no_tts:
            return
        audio = tts(self.api_key, text, self.args.voice, self.args.tts_language)
        if audio:
            print("[TTS] playing (%s, %s)" % (self.args.voice, self.args.tts_language), flush=True)
            play_wav(audio)

    def poll_for_stop(self):
        """While walking: Enter -> listen; send a stop if one was spoken."""
        if msvcrt is None or not msvcrt.kbhit():
            return
        if msvcrt.getwch() not in ("\r", "\n"):
            return
        transcript = self.listen()
        banner("TRANSCRIPT", repr(transcript))
        parsed = command_parser.parse(transcript or "")
        banner("PARSED", parsed)
        if parsed["intent"] == "stop":
            cmd = make_command(parsed, self.mode)
            banner("COMMAND", "%s (the walk's result is the reply)" % cmd)
            self.robot.publish(transcript, cmd)
        else:
            print("[BUSY] Gurdy is walking; only a stop works now", flush=True)

    def handle(self, transcript):
        banner("TRANSCRIPT", repr(transcript))
        parsed = command_parser.parse(transcript)
        banner("PARSED", parsed)
        cmd = make_command(parsed, self.mode)
        feedback = None
        if parsed["intent"] == "mode":
            self.mode = parsed["mode"]
            banner("MODE", "walks now use %s" % self.mode)
        if cmd is None:
            self.robot.publish(transcript=transcript)
        else:
            banner("COMMAND", cmd)
            walking = cmd["action"] == "walk_forward"
            if walking and msvcrt is not None:
                print("[ROBOT] walking... press Enter to speak a stop", flush=True)
            feedback = self.robot.send_and_wait(
                cmd, transcript=transcript,
                timeout=WALK_TIMEOUT if walking else REPLY_TIMEOUT,
                poll=self.poll_for_stop if walking else None)
            banner("ROBOT", feedback if feedback is not None else "no reply")
        self.say(replies.build_reply(parsed, feedback, self.mode))

    def run(self):
        print("Gurdy voice demo. Mode: %s. Container: %s%s" % (
            self.mode, self.args.container, " (dry run)" if self.args.dry_run else ""))
        while True:
            try:
                key = input("\nPress Enter to talk (q + Enter to quit): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if key == "q":
                break
            transcript = self.listen()
            if transcript is None:
                continue  # STT failed; already printed
            if not transcript:
                banner("TRANSCRIPT", "(empty: nothing heard)")
                continue
            self.handle(transcript)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=4.0, help="recording length")
    ap.add_argument("--language", default="hi-IN", help="STT language_code")
    ap.add_argument("--voice", default="Poorvi", help="Timbre v2.5 voice")
    ap.add_argument("--tts-language", default="hi-en", help="TTS language")
    ap.add_argument("--mode", choices=("scripted", "rl"), default="scripted",
                    help="walk mode at start; say 'RL mode' / 'normal mode' to switch")
    ap.add_argument("--container", default=DEFAULT_CONTAINER)
    ap.add_argument("--setup", default=DEFAULT_SETUP, help="setup.bash in the container")
    ap.add_argument("--echo-delay", type=float, default=2.0,
                    help="s to let rostopic echo subscribe before publishing")
    ap.add_argument("--type", action="store_true", help="type transcripts instead of speaking")
    ap.add_argument("--no-tts", action="store_true", help="print replies only")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the docker commands instead of running them")
    VoiceDemo(ap.parse_args()).run()


if __name__ == "__main__":
    main()
