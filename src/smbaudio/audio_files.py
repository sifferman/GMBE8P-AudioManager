import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy
import soundfile
import soxr

from smbaudio.errors import AudioManagerError

LARGEST_SAMPLE_VALUE = 0x7FFF
EXTENSIONS_READABLE_WITHOUT_FFMPEG = (".wav", ".flac")


def to_integer_samples(samples: numpy.ndarray) -> numpy.ndarray:
    scaled = numpy.round(samples.astype(numpy.float64) * LARGEST_SAMPLE_VALUE)
    return numpy.clip(scaled, -LARGEST_SAMPLE_VALUE, LARGEST_SAMPLE_VALUE).astype(numpy.int16)


def same_sound_in_both_channels(two_channel: numpy.ndarray) -> numpy.ndarray:
    mixed = numpy.round(two_channel.astype(numpy.float64).mean(axis=1, keepdims=True)).astype(numpy.int16)
    return numpy.repeat(mixed, 2, axis=1)


def fitted_to_sample_count(samples: numpy.ndarray, sample_count: int) -> numpy.ndarray:
    missing = numpy.zeros((max(sample_count - len(samples), 0), *samples.shape[1:]), dtype=samples.dtype)
    return numpy.concatenate([samples[:sample_count], missing])


@dataclass(frozen=True)
class LoadedAudio:
    samples: numpy.ndarray
    sample_rate: int

    @classmethod
    def read_with_soundfile(cls, audio_path: Path) -> "LoadedAudio":
        samples, sample_rate = soundfile.read(str(audio_path), dtype="float32", always_2d=True)
        return cls(samples, sample_rate)

    @classmethod
    def read_with_ffmpeg(cls, audio_path: Path) -> "LoadedAudio":
        with tempfile.TemporaryDirectory() as temporary_directory:
            converted_path = Path(temporary_directory) / "converted.wav"
            conversion = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(audio_path), "-vn", "-c:a", "pcm_f32le", str(converted_path)],
                capture_output=True,
                check=False,
                text=True,
            )
            if conversion.returncode != 0:
                raise AudioManagerError(f"ffmpeg could not read {audio_path}: {conversion.stderr.strip()}")
            return cls.read_with_soundfile(converted_path)

    @property
    def channel_count(self) -> int:
        return self.samples.shape[1]

    def resampled(self, sample_rate: int) -> "LoadedAudio":
        if self.sample_rate == sample_rate:
            return self
        return LoadedAudio(soxr.resample(self.samples, self.sample_rate, sample_rate, quality="VHQ"), sample_rate)

    def mixed_to_single_channel(self) -> "LoadedAudio":
        return LoadedAudio(self.samples.mean(axis=1, keepdims=True), self.sample_rate)

    def single_channel_samples(self, sample_rate: int) -> numpy.ndarray:
        return to_integer_samples(self.resampled(sample_rate).mixed_to_single_channel().samples[:, 0])

    def two_channel_samples(self, sample_rate: int) -> numpy.ndarray:
        resampled = self.resampled(sample_rate)
        if resampled.channel_count == 2:
            return to_integer_samples(resampled.samples)
        return to_integer_samples(numpy.repeat(resampled.mixed_to_single_channel().samples, 2, axis=1))


def is_ffmpeg_installed() -> bool:
    return shutil.which("ffmpeg") is not None


@dataclass(frozen=True)
class ReplacementAudioFile:
    path_written_in_config: str
    path: Path

    def __str__(self) -> str:
        return self.path_written_in_config

    @property
    def is_readable_without_ffmpeg(self) -> bool:
        return self.path.suffix.lower() in EXTENSIONS_READABLE_WITHOUT_FFMPEG

    @property
    def reason_it_cannot_be_read(self) -> str | None:
        if not self.path.is_file():
            return f"replacement file does not exist: {self}"
        if not self.is_readable_without_ffmpeg and not is_ffmpeg_installed():
            return f"{self} needs ffmpeg installed; without ffmpeg only WAV and FLAC can be read"
        return None

    def read(self) -> LoadedAudio:
        if self.is_readable_without_ffmpeg:
            return LoadedAudio.read_with_soundfile(self.path)
        return LoadedAudio.read_with_ffmpeg(self.path)


def write_wav_file(wav_path: Path, samples: numpy.ndarray, sample_rate: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(wav_path), samples, sample_rate, subtype="PCM_16")
