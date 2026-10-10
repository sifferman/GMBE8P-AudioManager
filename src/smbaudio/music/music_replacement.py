import struct
from dataclasses import dataclass

import numpy

from smbaudio.audio_files import ReplacementAudioFile, fitted_to_sample_count, same_sound_in_both_channels
from smbaudio.disc.dol_executable import ExecutablePatch
from smbaudio.music.dtk_adpcm import CHANNEL_COUNT, SAMPLE_RATE, encode_dtk_adpcm, encoded_size_for_samples
from smbaudio.music.music_catalog import MusicCatalog, MusicFile
from smbaudio.size_change import SizeChange

SAMPLE_COUNT_KEPT_AFTER_PLAYED_SAMPLES = SAMPLE_RATE
SILENCE = numpy.zeros((0, CHANNEL_COUNT), dtype=numpy.int16)


def _read_music(replacement: ReplacementAudioFile) -> numpy.ndarray:
    return replacement.read().two_channel_samples(SAMPLE_RATE)


@dataclass(frozen=True)
class PlannedMusicFile:
    music_file: MusicFile
    played_samples: numpy.ndarray
    source_description: str

    @classmethod
    def replaced_by(
        cls, music_file: MusicFile, replacement: ReplacementAudioFile, played_samples: numpy.ndarray
    ) -> "PlannedMusicFile":
        return cls(music_file, played_samples, f"replaced: {replacement}")

    @classmethod
    def generated_from(
        cls, music_file: MusicFile, replacement: ReplacementAudioFile, played_samples: numpy.ndarray
    ) -> "PlannedMusicFile":
        return cls(music_file, played_samples, f"generated from: {replacement}")

    @property
    def played_sample_count(self) -> int:
        return len(self.played_samples)

    @property
    def encoded_size(self) -> int:
        return encoded_size_for_samples(self.played_sample_count + SAMPLE_COUNT_KEPT_AFTER_PLAYED_SAMPLES)

    @property
    def size_change(self) -> SizeChange:
        return SizeChange(
            self.music_file.encoded_name, self.music_file.disc_file.size, self.encoded_size, self.source_description
        )

    def played_sample_count_patch(self, table_index: int, file_offset: int) -> ExecutablePatch:
        return ExecutablePatch(
            file_offset,
            struct.pack(">I", self.played_sample_count),
            f"music track {table_index} length: "
            f"{self.music_file.played_sample_count} -> {self.played_sample_count} samples",
        )


def _plan_both_versions(
    surround_file: MusicFile,
    mono_file: MusicFile,
    surround_replacement: ReplacementAudioFile | None,
    mono_replacement: ReplacementAudioFile | None,
) -> list[PlannedMusicFile]:
    if surround_replacement is None:
        mono_samples = _read_music(mono_replacement)
        return [
            PlannedMusicFile.generated_from(surround_file, mono_replacement, mono_samples),
            PlannedMusicFile.replaced_by(mono_file, mono_replacement, mono_samples),
        ]
    surround_samples = _read_music(surround_replacement)
    planned_surround_file = PlannedMusicFile.replaced_by(surround_file, surround_replacement, surround_samples)
    if mono_replacement is None:
        mono_samples = same_sound_in_both_channels(surround_samples)
        return [planned_surround_file, PlannedMusicFile.generated_from(mono_file, surround_replacement, mono_samples)]
    mono_samples = fitted_to_sample_count(_read_music(mono_replacement), len(surround_samples))
    return [planned_surround_file, PlannedMusicFile.replaced_by(mono_file, mono_replacement, mono_samples)]


class MusicReplacementPlan:
    def __init__(self, music_catalog: MusicCatalog, replacements: dict[str, ReplacementAudioFile]):
        self._music_catalog = music_catalog
        self.planned_files = [
            planned_file
            for surround_file in music_catalog.music_files.values()
            if not surround_file.is_mono_version
            for planned_file in self._planned_versions(surround_file, replacements)
        ]
        self._planned_file_by_dumped_name = {
            planned_file.music_file.dumped_name: planned_file for planned_file in self.planned_files
        }

    @property
    def encoded_size_by_disc_path(self) -> dict[str, int]:
        return {planned_file.music_file.disc_path: planned_file.encoded_size for planned_file in self.planned_files}

    @property
    def empty_replacement_problems(self) -> list[str]:
        return [
            f"replacement contains no audio ({planned_file.source_description})"
            for planned_file in self.planned_files
            if planned_file.played_sample_count == 0
        ]

    @property
    def executable_patches(self) -> list[ExecutablePatch]:
        track_table = self._music_catalog.track_table
        patch_by_file_offset = {}
        for planned_file in self.planned_files:
            for table_index in planned_file.music_file.table_indices_holding_played_sample_count:
                file_offset = track_table.played_sample_count_file_offset(table_index)
                patch_by_file_offset[file_offset] = planned_file.played_sample_count_patch(table_index, file_offset)
        return [patch_by_file_offset[file_offset] for file_offset in sorted(patch_by_file_offset)]

    def encoded(self, planned_file: PlannedMusicFile) -> bytes:
        samples_kept_afterwards = fitted_to_sample_count(
            self._music_played_after(planned_file.music_file), SAMPLE_COUNT_KEPT_AFTER_PLAYED_SAMPLES
        )
        return encode_dtk_adpcm(numpy.concatenate([planned_file.played_samples, samples_kept_afterwards]))

    def _planned_versions(
        self, surround_file: MusicFile, replacements: dict[str, ReplacementAudioFile]
    ) -> list[PlannedMusicFile]:
        mono_file = self._music_catalog.other_version_of(surround_file)
        surround_replacement = replacements.get(surround_file.dumped_name)
        mono_replacement = replacements.get(mono_file.dumped_name)
        if surround_replacement is None and mono_replacement is None:
            return []
        return _plan_both_versions(surround_file, mono_file, surround_replacement, mono_replacement)

    def _music_played_after(self, music_file: MusicFile) -> numpy.ndarray:
        file_played_afterwards = self._music_catalog.file_played_after(music_file)
        if file_played_afterwards is None:
            return SILENCE
        if file_played_afterwards.dumped_name in self._planned_file_by_dumped_name:
            return self._planned_file_by_dumped_name[file_played_afterwards.dumped_name].played_samples
        return file_played_afterwards.decoded_first_samples(
            self._music_catalog.disc_image, SAMPLE_COUNT_KEPT_AFTER_PLAYED_SAMPLES
        )
