"""STT for a dictation session: the recording's audio goes to Muse Voice Transcribe once, and comes back as text + turns."""

import io
import json
import subprocess
import wave

import pytest

import asr

# The complete response schema of POST /v1/asr/transcribe (mode ENDPOINTING), as captured from a live call.
RESPONSE = {
    "sessionId": "46db98e5-7611-4f4c-84ec-98da933d3369",
    "transcript": "turn on the lights. yes. open the terminal",
    "audioDurationMs": 6080,
    "turns": [
        {"turnId": 0, "startMs": 120, "endMs": 1840, "transcript": "turn on the lights."},
        {"turnId": 1, "startMs": 2400, "endMs": 2900, "transcript": "yes."},
        {"turnId": 2, "startMs": 3500, "endMs": 5960, "transcript": "open the terminal"},
    ],
}


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> bytes:
    """2 s of webm with a 48 kHz stereo audio track, so the 16 kHz mono conversion is real work."""
    path = tmp_path_factory.mktemp("asr") / "recording.webm"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30", "-f", "lavfi",
                    "-i", "sine=sample_rate=48000", "-t", "2", "-ac", "2", "-c:v", "libvpx", "-c:a", "libopus", path],
                   check=True)
    return path.read_bytes()


class FakeASR:
    """Stands in for httpx.post against the ASR endpoint: records the request, answers with RESPONSE."""

    def __init__(self, body=RESPONSE):
        self.body, self.calls = body, []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        body = self.body

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return body

        return Response()


@pytest.fixture
def fake_asr(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    fake = FakeASR()
    monkeypatch.setattr(asr.httpx, "post", fake)
    return fake


def test_transcribe_turns_returns_the_transcript_and_each_turn_in_ms(fake_asr, recording):
    assert asr.transcribe_turns(recording) == {
        "text": "turn on the lights. yes. open the terminal",
        "turns": [
            {"start_ms": 120, "end_ms": 1840, "text": "turn on the lights."},
            {"start_ms": 2400, "end_ms": 2900, "text": "yes."},
            {"start_ms": 3500, "end_ms": 5960, "text": "open the terminal"},
        ],
    }


def test_transcribe_turns_sends_one_endpointing_request_with_16khz_mono_pcm_wav(fake_asr, recording):
    asr.transcribe_turns(recording)

    ((url, kwargs),) = fake_asr.calls
    assert url.endswith("/asr/transcribe")
    request = json.loads(kwargs["files"]["request"][1])
    assert request == {"mode": "ENDPOINTING", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    with wave.open(io.BytesIO(kwargs["files"]["audio"][1])) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert audio.getnframes() / 16000 == pytest.approx(2.0, abs=0.05)  # the whole recording's audio, nothing cut
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_transcribe_is_the_text_alone_so_the_server_can_switch_to_it(fake_asr, recording):
    assert asr.transcribe(recording) == "turn on the lights. yes. open the terminal"
