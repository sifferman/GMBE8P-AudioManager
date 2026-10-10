import json

import numpy
import pytest
import soundfile
from conftest import tone_with_noise

from smbaudio.errors import InvalidReplacementsError
from smbaudio.replacement_config import ReplacementConfig

SAMPLE_RATE = 22050


def write_config(tmp_path, config: dict) -> ReplacementConfig:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return ReplacementConfig(tmp_path / "config.json")


def test_path_alone_replaces_with_the_whole_file(tmp_path):
    replacement = write_config(tmp_path, {"music/st1.wav": "songs/cool.wav"}).replacements["music/st1.wav"]
    assert replacement.path == tmp_path / "songs" / "cool.wav"
    assert (replacement.time_start, replacement.time_stop) == (0, None)


def test_time_slot_replaces_with_only_that_part_of_the_file(tmp_path):
    whole_file = tone_with_noise(SAMPLE_RATE * 3, SAMPLE_RATE, 500, seed=7)
    soundfile.write(tmp_path / "mix.wav", whole_file, SAMPLE_RATE)
    config = write_config(
        tmp_path,
        {
            "sfx/comn/first.wav": {"replacement_file": "mix.wav", "time_start": 0.5, "time_stop": 2},
            "sfx/comn/second.wav": {"replacement_file": "mix.wav", "time_start": 2.5},
        },
    )
    first_slot = config.replacements["sfx/comn/first.wav"].read().single_channel_samples(SAMPLE_RATE)
    second_slot = config.replacements["sfx/comn/second.wav"].read().single_channel_samples(SAMPLE_RATE)
    assert numpy.array_equal(first_slot, whole_file[SAMPLE_RATE // 2 : SAMPLE_RATE * 2])
    assert numpy.array_equal(second_slot, whole_file[SAMPLE_RATE * 5 // 2 :])


def test_missing_file_shared_by_many_entries_is_reported_once(tmp_path):
    config = write_config(
        tmp_path,
        {
            "sfx/comn/first.wav": {"replacement_file": "mix.wav", "time_stop": 1},
            "sfx/comn/second.wav": {"replacement_file": "mix.wav", "time_start": 1},
        },
    )
    with pytest.raises(InvalidReplacementsError) as raised:
        config.replacements_checked_against({"sfx/comn/first.wav", "sfx/comn/second.wav"})
    assert len(raised.value.problems) == 1


@pytest.mark.parametrize(
    "entry",
    [
        5,
        {"time_start": 1},
        {"replacement_file": "mix.wav", "time_start": 2, "time_stop": 1},
        {"replacement_file": "mix.wav", "time_start": -1},
        {"replacement_file": "mix.wav", "volume": 2},
    ],
)
def test_malformed_entries_are_rejected(tmp_path, entry):
    with pytest.raises(InvalidReplacementsError):
        write_config(tmp_path, {"music/st1.wav": entry})
