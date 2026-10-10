from dataclasses import dataclass, replace

import numpy

from smbaudio.audio_files import ReplacementAudioFile
from smbaudio.size_change import SizeChange
from smbaudio.sound_effects.dsp_adpcm import encoded_size_for_samples
from smbaudio.sound_effects.dsp_adpcm_encoder import encode_dsp_adpcm
from smbaudio.sound_effects.musyx_files import DirectorySample, SampleCoding, SampleDirectory
from smbaudio.sound_effects.sound_bank import SoundBank

AUDIO_MEMORY_BUDGET = 0x700000
LOADED_SIZE_MULTIPLE = 32
BANKS_ALWAYS_IN_AUDIO_MEMORY = ("allse", "comn")
SAMPLE_DATA_ALIGNMENT = 32


class AudioMemoryUse:
    def __init__(self, sample_data_size_by_bank: dict[str, int]):
        self._sample_data_size_by_bank = sample_data_size_by_bank

    @property
    def problems(self) -> list[str]:
        return sorted(
            {
                self._problem(banks_loaded_together)
                for banks_loaded_together in map(self._banks_loaded_with, self._sample_data_size_by_bank)
                if self._needed_size(banks_loaded_together) >= AUDIO_MEMORY_BUDGET
            }
        )

    @staticmethod
    def _banks_loaded_with(bank_name: str) -> tuple[str, ...]:
        return tuple(sorted({*BANKS_ALWAYS_IN_AUDIO_MEMORY, bank_name}))

    def _loaded_size(self, bank_name: str) -> int:
        sample_data_size = self._sample_data_size_by_bank[bank_name]
        return sample_data_size + -sample_data_size % LOADED_SIZE_MULTIPLE

    def _needed_size(self, banks_loaded_together: tuple[str, ...]) -> int:
        return sum(self._loaded_size(bank_name) for bank_name in banks_loaded_together)

    def _problem(self, banks_loaded_together: tuple[str, ...]) -> str:
        needed_size = self._needed_size(banks_loaded_together)
        return (
            f"sound effects in {' + '.join(banks_loaded_together)} need {needed_size} bytes of audio memory (ARAM) "
            f"but must stay under {AUDIO_MEMORY_BUDGET}; "
            f"shorten the replacements by {needed_size - AUDIO_MEMORY_BUDGET + 1} bytes"
        )


def _aligned_up(offset: int) -> int:
    return offset + -offset % SAMPLE_DATA_ALIGNMENT


@dataclass(frozen=True)
class EncodedSample:
    directory_sample: DirectorySample
    encoded_bytes: bytes


@dataclass(frozen=True)
class RebuiltSoundBank:
    sample_directory_bytes: bytes
    sample_data_bytes: bytes


@dataclass(frozen=True)
class PlannedSample:
    original_sample: DirectorySample
    dumped_name: str
    samples: numpy.ndarray
    replacement: ReplacementAudioFile

    @classmethod
    def read(
        cls, original_sample: DirectorySample, dumped_name: str, replacement: ReplacementAudioFile
    ) -> "PlannedSample":
        samples = replacement.read().single_channel_samples(original_sample.sample_rate)
        return cls(original_sample, dumped_name, samples, replacement)

    @property
    def encoded_size(self) -> int:
        return encoded_size_for_samples(len(self.samples))

    @property
    def size_change(self) -> SizeChange:
        return SizeChange(
            self.dumped_name, self.original_sample.encoded_size, self.encoded_size, f"replaced: {self.replacement}"
        )

    def encoded(self) -> EncodedSample:
        encoded = encode_dsp_adpcm(self.samples)
        first_frame_header = encoded.frame_bytes[0]
        is_looping = self.original_sample.is_looping
        return EncodedSample(
            replace(
                self.original_sample,
                sample_count=encoded.sample_count,
                loop_start_sample=0,
                loop_sample_count=encoded.sample_count if is_looping else 0,
                coding=SampleCoding(
                    first_frame_header, first_frame_header if is_looping else 0, 0, 0, encoded.coefficients
                ),
            ),
            encoded.frame_bytes,
        )


@dataclass(frozen=True)
class PlannedSoundBank:
    sound_bank: SoundBank
    planned_samples: tuple[PlannedSample, ...]

    @classmethod
    def plan(cls, sound_bank: SoundBank, replacements: dict[str, ReplacementAudioFile]) -> "PlannedSoundBank":
        return cls(
            sound_bank,
            tuple(
                PlannedSample.read(sample, sound_bank.dumped_name(sample), replacements[sound_bank.dumped_name(sample)])
                for sample in sound_bank.samples
                if sound_bank.dumped_name(sample) in replacements
            ),
        )

    @property
    def is_changed(self) -> bool:
        return bool(self.planned_samples)

    @property
    def sample_data_size(self) -> int:
        planned_size_by_sample_id = {
            planned.original_sample.sample_id: planned.encoded_size for planned in self.planned_samples
        }
        end_offset = 0
        for sample in self.sound_bank.samples_in_sample_data_order:
            end_offset = _aligned_up(end_offset) + planned_size_by_sample_id.get(sample.sample_id, sample.encoded_size)
        return end_offset

    @property
    def sample_data_size_change(self) -> SizeChange:
        return SizeChange(
            self.sound_bank.sample_data_disc_path,
            len(self.sound_bank.sample_data),
            self.sample_data_size,
            f"rebuilt, sound effects replaced: {len(self.planned_samples)}",
        )

    @property
    def empty_replacement_problems(self) -> list[str]:
        return [
            f"replacement contains no audio ({planned.replacement})"
            for planned in self.planned_samples
            if len(planned.samples) == 0
        ]

    def rebuilt(self) -> RebuiltSoundBank:
        encoded_sample_by_id = {
            sample.sample_id: EncodedSample(sample, self.sound_bank.encoded_bytes(sample))
            for sample in self.sound_bank.samples
        }
        for planned in self.planned_samples:
            encoded_sample_by_id[planned.original_sample.sample_id] = planned.encoded()

        sample_data = bytearray()
        relocated_sample_by_id = {}
        for original_sample in self.sound_bank.samples_in_sample_data_order:
            encoded_sample = encoded_sample_by_id[original_sample.sample_id]
            sample_data += bytes(_aligned_up(len(sample_data)) - len(sample_data))
            relocated_sample_by_id[original_sample.sample_id] = replace(
                encoded_sample.directory_sample, data_offset=len(sample_data)
            )
            sample_data += encoded_sample.encoded_bytes
        relocated_directory = SampleDirectory(
            tuple(relocated_sample_by_id[sample.sample_id] for sample in self.sound_bank.samples)
        )
        return RebuiltSoundBank(relocated_directory.serialized(), bytes(sample_data))
