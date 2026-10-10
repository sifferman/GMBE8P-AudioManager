import os
from pathlib import Path

import numpy
import pytest

RETAIL_ISO_ENVIRONMENT_VARIABLE = "SMBAUDIO_RETAIL_ISO"


@pytest.fixture(scope="session")
def retail_iso_path() -> Path:
    configured_path = os.environ.get(RETAIL_ISO_ENVIRONMENT_VARIABLE)
    if not configured_path or not Path(configured_path).is_file():
        pytest.skip(f"set {RETAIL_ISO_ENVIRONMENT_VARIABLE} to the retail ISO to run this test")
    return Path(configured_path)


@pytest.fixture(scope="session")
def retail_game_disc(retail_iso_path):
    from smbaudio.game_disc import GameDisc

    return GameDisc.open_verified(retail_iso_path)


def tone_with_noise(sample_count: int, sample_rate: int, frequency: float, seed: int) -> numpy.ndarray:
    time = numpy.arange(sample_count) / sample_rate
    noise = numpy.random.default_rng(seed).standard_normal(sample_count)
    return (9000 * numpy.sin(2 * numpy.pi * frequency * time) + 300 * noise).astype(numpy.int16)


def signal_to_noise_decibels(wanted: numpy.ndarray, actual: numpy.ndarray) -> float:
    wanted = wanted.astype(numpy.float64)
    noise = wanted - actual.astype(numpy.float64)
    return 10 * numpy.log10((wanted**2).sum() / max((noise**2).sum(), 1e-9))
