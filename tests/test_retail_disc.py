import json

import numpy
import pytest
import soundfile
from conftest import signal_to_noise_decibels, tone_with_noise

from smbaudio.disc.gamecube_disc_image import GameCubeDiscImage
from smbaudio.game_disc import RETAIL_DISC_IMAGE_SHA1, disc_image_sha1
from smbaudio.music.dtk_adpcm import decode_dtk_adpcm, encode_dtk_adpcm
from smbaudio.patch_command import patch_disc_image
from smbaudio.sound_effects.dsp_adpcm import decode_dsp_adpcm
from smbaudio.sound_effects.musyx_files import DirectorySample
from smbaudio.sound_effects.sound_bank import SoundBank, SoundBankReader
from smbaudio.sound_effects.sound_bank_rebuilding import PlannedSoundBank

REPLACED_BANK_NAME = "ending"
REPLACED_SOUND_EFFECT = "sfx/ending/SFXfly1_basic_macro.wav"


def sound_bank_named(sound_banks: list[SoundBank], bank_name: str) -> SoundBank:
    return next(sound_bank for sound_bank in sound_banks if sound_bank.name == bank_name)


def sample_named(sound_bank: SoundBank, dumped_name: str) -> DirectorySample:
    return next(sample for sample in sound_bank.samples if sound_bank.dumped_name(sample) == dumped_name)


def test_every_music_file_and_sound_effect_is_found(retail_game_disc):
    assert len(retail_game_disc.music_files) == 112
    assert len(retail_game_disc.sound_effect_names) == 601
    assert "music/selm_fight.wav" in retail_game_disc.music_files
    assert "sfx/comn/SFX1up_basic_macro.wav" in retail_game_disc.sound_effect_names


def test_rebuilding_sound_banks_without_replacements_reproduces_them(retail_game_disc):
    disc_image = retail_game_disc.disc_image
    for sound_bank in retail_game_disc.sound_banks:
        unchanged_bank = PlannedSoundBank(sound_bank, ())
        rebuilt = unchanged_bank.rebuilt()
        assert unchanged_bank.sample_data_size == len(sound_bank.sample_data)
        assert rebuilt.sample_data_bytes == sound_bank.sample_data
        assert rebuilt.sample_directory_bytes == disc_image.read_file(sound_bank.sample_directory_disc_path)


def test_encoding_decoded_original_music_reproduces_it(retail_game_disc):
    decoded = decode_dtk_adpcm(retail_game_disc.disc_image.read_file("test/snd/adp/ovr.adp"))
    assert signal_to_noise_decibels(decoded, decode_dtk_adpcm(encode_dtk_adpcm(decoded))) > 80


def test_patching_without_replacements_reproduces_the_retail_disc(retail_iso_path, tmp_path):
    (tmp_path / "config.json").write_text("{}")
    patch_disc_image(tmp_path / "config.json", retail_iso_path, tmp_path / "patched.iso")
    assert disc_image_sha1(tmp_path / "patched.iso") == RETAIL_DISC_IMAGE_SHA1


@pytest.fixture(scope="module")
def replacement_music() -> numpy.ndarray:
    return numpy.stack([tone_with_noise(96000, 48000, 440, seed=5)] * 2, axis=1)


@pytest.fixture(scope="module")
def replaced_sound_effect(retail_game_disc) -> DirectorySample:
    return sample_named(sound_bank_named(retail_game_disc.sound_banks, REPLACED_BANK_NAME), REPLACED_SOUND_EFFECT)


@pytest.fixture(scope="module")
def replacement_sound_effect(replaced_sound_effect) -> numpy.ndarray:
    return tone_with_noise(20000, replaced_sound_effect.sample_rate, 900, seed=6)


@pytest.fixture(scope="module")
def patched_disc(retail_iso_path, replacement_music, replaced_sound_effect, replacement_sound_effect, tmp_path_factory):
    work_directory = tmp_path_factory.mktemp("patched_disc")
    soundfile.write(work_directory / "song.wav", replacement_music, 48000)
    sample_rate = replaced_sound_effect.sample_rate
    other_sound_before = tone_with_noise(sample_rate, sample_rate, 200, seed=7)
    other_sound_after = tone_with_noise(sample_rate, sample_rate, 300, seed=8)
    soundfile.write(
        work_directory / "mix.flac",
        numpy.concatenate([other_sound_before, replacement_sound_effect, other_sound_after]),
        sample_rate,
    )
    sound_effect_time_slot = {
        "replacement_file": "mix.flac",
        "time_start": len(other_sound_before) / sample_rate,
        "time_stop": (len(other_sound_before) + len(replacement_sound_effect)) / sample_rate,
    }
    (work_directory / "config.json").write_text(
        json.dumps({"music/ovr.wav": "song.wav", REPLACED_SOUND_EFFECT: sound_effect_time_slot})
    )
    patch_disc_image(work_directory / "config.json", retail_iso_path, work_directory / "patched.iso")
    return GameCubeDiscImage(work_directory / "patched.iso")


@pytest.fixture(scope="module")
def patched_sound_bank(patched_disc) -> SoundBank:
    return SoundBankReader(patched_disc, patched_disc.read_executable()).read_sound_bank(REPLACED_BANK_NAME)


def test_patched_disc_holds_the_replaced_music_in_both_versions(patched_disc, replacement_music):
    for music_path in ("test/snd/adp/ovr.adp", "test/snd/adp/ovrm.adp"):
        music_file = patched_disc.find_file(music_path)
        assert music_file.offset % 0x8000 == 0 and music_file.size % 0x8000 == 0
        decoded = decode_dtk_adpcm(patched_disc.read_file(music_path))[: len(replacement_music)]
        assert signal_to_noise_decibels(replacement_music, decoded) > 30


def test_patched_disc_holds_the_replaced_sound_effect(
    patched_sound_bank, replaced_sound_effect, replacement_sound_effect
):
    patched_sample = sample_named(patched_sound_bank, REPLACED_SOUND_EFFECT)
    assert patched_sample.sample_count == len(replacement_sound_effect)
    assert patched_sample.sample_rate == replaced_sound_effect.sample_rate
    assert patched_sample.is_looping == replaced_sound_effect.is_looping
    decoded = decode_dsp_adpcm(patched_sound_bank.encoded_sample(patched_sample))
    assert signal_to_noise_decibels(replacement_sound_effect, decoded) > 30


def test_patched_disc_keeps_the_other_sound_effects_in_the_bank(retail_game_disc, patched_sound_bank):
    retail_sound_bank = sound_bank_named(retail_game_disc.sound_banks, REPLACED_BANK_NAME)
    assert patched_sound_bank.dumped_names == retail_sound_bank.dumped_names
    for retail_sample in retail_sound_bank.samples:
        if retail_sound_bank.dumped_name(retail_sample) != REPLACED_SOUND_EFFECT:
            patched_sample = sample_named(patched_sound_bank, retail_sound_bank.dumped_name(retail_sample))
            assert patched_sound_bank.encoded_sample(patched_sample) == retail_sound_bank.encoded_sample(retail_sample)


def test_patching_leaves_the_input_disc_untouched(patched_disc, retail_iso_path):
    assert disc_image_sha1(retail_iso_path) == RETAIL_DISC_IMAGE_SHA1
