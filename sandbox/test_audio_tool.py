from __future__ import annotations

import argparse
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None


DEFAULT_MODEL = "small"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SECONDS = 5


def load_model(model_size: str) -> WhisperModel:
    """Load Whisper with a simple GPU->CPU fallback."""
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print("Whisper backend: GPU")
        return model
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Whisper backend: CPU")
        return model


def transcribe_audio(model: WhisperModel, audio_path: Path, beam_size: int) -> str:
    """Transcribe one audio file and print useful debug output."""
    segments, info = model.transcribe(str(audio_path), beam_size=beam_size)

    print(
        f"Detected language: {info.language} "
        f"({info.language_probability:.0%})"
    )

    texts: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        texts.append(text)
        print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {text}")

    transcript = " ".join(texts).strip()
    print("\nTranscript:")
    print(transcript or "<empty>")
    return transcript


def record_to_wav(
    output_path: Path,
    *,
    seconds: int,
    sample_rate: int,
) -> Path:
    """Record microphone audio to a WAV file."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed. Install it first to record from microphone."
        )
    if wavfile is None:
        raise RuntimeError(
            "scipy is not installed. Install it first to save recorded audio."
        )

    print(f"Recording for {seconds} seconds. Speak now...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Recording complete.")

    wav_data = np.squeeze(audio)
    wavfile.write(output_path, sample_rate, wav_data)
    print(f"Saved recording to: {output_path}")
    return output_path


def record_until_enter(
    output_path: Path,
    *,
    sample_rate: int,
) -> Path:
    """Record microphone audio until the user presses Enter again."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed. Install it first to record from microphone."
        )
    if wavfile is None:
        raise RuntimeError(
            "scipy is not installed. Install it first to save recorded audio."
        )

    print("Press Enter to start recording.")
    input()
    print("Recording... press Enter to stop.")

    captured_chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def callback(indata, frames, time, status) -> None:
        del frames, time
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        captured_chunks.append(indata.copy())
        if stop_event.is_set():
            raise sd.CallbackStop()

    stop_thread = threading.Thread(target=input, daemon=True)
    stop_thread.start()

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        while not stop_event.is_set():
            if not stop_thread.is_alive():
                stop_event.set()
                break
            sd.sleep(100)

    if not captured_chunks:
        raise RuntimeError("No audio was captured from the microphone.")

    audio = np.concatenate(captured_chunks, axis=0)
    wav_data = np.squeeze(audio)
    wavfile.write(output_path, sample_rate, wav_data)
    print("Recording complete.")
    print(f"Saved recording to: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Whisper CLI for testing microphone and audio-file transcription."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to an existing audio file to transcribe.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record from the microphone in the terminal session before transcribing.",
    )
    parser.add_argument(
        "--record-live",
        action="store_true",
        help="Press Enter to start recording and press Enter again to stop.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=DEFAULT_SECONDS,
        help=f"Recording duration in seconds when using --record (default: {DEFAULT_SECONDS}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Microphone sample rate (default: {DEFAULT_SAMPLE_RATE}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model size to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size passed to faster-whisper transcribe().",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    selected_modes = sum(
        bool(mode)
        for mode in (args.record, args.record_live, args.input_file)
    )
    if selected_modes == 0:
        parser.error("Use one of: --record, --record-live, or --input-file PATH.")

    if selected_modes > 1:
        parser.error(
            "Use only one mode at a time: --record, --record-live, or --input-file PATH."
        )

    model = load_model(args.model)

    if args.input_file:
        audio_path = args.input_file.expanduser().resolve()
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}", file=sys.stderr)
            return 1
        transcribe_audio(model, audio_path, args.beam_size)
        return 0

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        if args.record_live:
            record_until_enter(
                temp_path,
                sample_rate=args.sample_rate,
            )
        else:
            record_to_wav(
                temp_path,
                seconds=args.seconds,
                sample_rate=args.sample_rate,
            )
        transcribe_audio(model, temp_path, args.beam_size)
        return 0
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
