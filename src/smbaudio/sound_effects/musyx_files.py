import struct
from collections.abc import Iterator
from dataclasses import dataclass

from smbaudio.sound_effects.dsp_adpcm import FRAME_SIZE, encoded_size_for_samples

SAMPLE_ENTRY_FORMAT = ">H2xI4xBxHIIII"
SAMPLE_ENTRY_SIZE = struct.calcsize(SAMPLE_ENTRY_FORMAT)
END_OF_LIST = b"\xff\xff\xff\xff"
SAMPLE_CODING_FORMAT = ">HBBhh16h"
SAMPLE_CODING_SIZE = struct.calcsize(SAMPLE_CODING_FORMAT)
SAMPLE_COUNT_MASK = 0xFFFFFF
GROUP_HEADER_FORMAT = ">IHH5II"
SOUND_EFFECT_GROUP_TYPE = 1
SOUND_EFFECT_COUNT_SIZE = 4
SOUND_EFFECT_ENTRY_FORMAT = ">HH6x"
SOUND_EFFECT_ENTRY_SIZE = struct.calcsize(SOUND_EFFECT_ENTRY_FORMAT)
MACRO_HEADER_FORMAT = ">IH2x"
MACRO_HEADER_SIZE = struct.calcsize(MACRO_HEADER_FORMAT)
MACRO_COMMAND_SIZE = 8
START_SAMPLE_COMMAND = 0x10


def _is_end_of_list(file_bytes: bytes, offset: int) -> bool:
    return file_bytes[offset : offset + len(END_OF_LIST)] == END_OF_LIST


@dataclass(frozen=True)
class SampleCoding:
    first_frame_header: int
    loop_frame_header: int
    loop_oldest_history: int
    loop_newest_history: int
    coefficients: tuple[int, ...]

    @classmethod
    def parse(cls, directory_bytes: bytes, coding_offset: int) -> "SampleCoding":
        _, first_frame_header, loop_frame_header, loop_oldest_history, loop_newest_history, *coefficients = (
            struct.unpack_from(SAMPLE_CODING_FORMAT, directory_bytes, coding_offset)
        )
        return cls(first_frame_header, loop_frame_header, loop_oldest_history, loop_newest_history, tuple(coefficients))

    def serialized(self) -> bytes:
        return struct.pack(
            SAMPLE_CODING_FORMAT,
            FRAME_SIZE,
            self.first_frame_header,
            self.loop_frame_header,
            self.loop_oldest_history,
            self.loop_newest_history,
            *self.coefficients,
        )


@dataclass(frozen=True)
class DirectorySample:
    sample_id: int
    data_offset: int
    base_note: int
    sample_rate: int
    sample_count: int
    loop_start_sample: int
    loop_sample_count: int
    coding: SampleCoding

    @classmethod
    def parse(cls, directory_bytes: bytes, entry_offset: int) -> "DirectorySample":
        (
            sample_id,
            data_offset,
            base_note,
            sample_rate,
            sample_count,
            loop_start_sample,
            loop_sample_count,
            coding_offset,
        ) = struct.unpack_from(SAMPLE_ENTRY_FORMAT, directory_bytes, entry_offset)
        return cls(
            sample_id,
            data_offset,
            base_note,
            sample_rate,
            sample_count & SAMPLE_COUNT_MASK,
            loop_start_sample,
            loop_sample_count,
            SampleCoding.parse(directory_bytes, coding_offset),
        )

    @property
    def is_looping(self) -> bool:
        return self.loop_sample_count > 0

    @property
    def encoded_size(self) -> int:
        return encoded_size_for_samples(self.sample_count)

    def serialized_entry(self, coding_offset: int) -> bytes:
        return struct.pack(
            SAMPLE_ENTRY_FORMAT,
            self.sample_id,
            self.data_offset,
            self.base_note,
            self.sample_rate,
            self.sample_count,
            self.loop_start_sample,
            self.loop_sample_count,
            coding_offset,
        )


@dataclass(frozen=True)
class SampleDirectory:
    samples: tuple[DirectorySample, ...]

    @classmethod
    def parse(cls, directory_bytes: bytes) -> "SampleDirectory":
        samples = []
        entry_offset = 0
        while not _is_end_of_list(directory_bytes, entry_offset):
            samples.append(DirectorySample.parse(directory_bytes, entry_offset))
            entry_offset += SAMPLE_ENTRY_SIZE
        return cls(tuple(samples))

    def serialized(self) -> bytes:
        first_coding_offset = len(self.samples) * SAMPLE_ENTRY_SIZE + len(END_OF_LIST)
        entries = b"".join(
            sample.serialized_entry(first_coding_offset + sample_index * SAMPLE_CODING_SIZE)
            for sample_index, sample in enumerate(self.samples)
        )
        codings = b"".join(sample.coding.serialized() for sample in self.samples)
        return entries + END_OF_LIST + codings


@dataclass(frozen=True)
class SoundMacro:
    macro_id: int
    command_bytes: bytes

    @property
    def started_sample_id(self) -> int | None:
        for command_offset in range(0, len(self.command_bytes), MACRO_COMMAND_SIZE):
            sample_id, command = struct.unpack_from(">xHB", self.command_bytes, command_offset)
            if command == START_SAMPLE_COMMAND:
                return sample_id
        return None


class SoundMacroPool:
    def __init__(self, pool_bytes: bytes):
        self._pool_bytes = pool_bytes
        self.sample_id_by_macro_id: dict[int, int] = {}
        for macro in self._macros():
            if macro.started_sample_id is not None:
                self.sample_id_by_macro_id.setdefault(macro.macro_id, macro.started_sample_id)

    def _macros(self) -> Iterator[SoundMacro]:
        macro_offset = struct.unpack_from(">I", self._pool_bytes)[0]
        while not _is_end_of_list(self._pool_bytes, macro_offset):
            macro_size, macro_id = struct.unpack_from(MACRO_HEADER_FORMAT, self._pool_bytes, macro_offset)
            yield SoundMacro(macro_id, self._pool_bytes[macro_offset + MACRO_HEADER_SIZE : macro_offset + macro_size])
            macro_offset += macro_size


class SoundEffectProject:
    def __init__(self, project_bytes: bytes):
        self._project_bytes = project_bytes
        self.macro_id_by_sound_effect_id: dict[int, int] = {}
        for table_offset in self._sound_effect_table_offsets():
            self.macro_id_by_sound_effect_id.update(self._read_sound_effect_table(table_offset))

    def _sound_effect_table_offsets(self) -> Iterator[int]:
        group_offset = 0
        while not _is_end_of_list(self._project_bytes, group_offset):
            following_group_offset, _, group_type, *_, sound_effect_table_offset = struct.unpack_from(
                GROUP_HEADER_FORMAT, self._project_bytes, group_offset
            )
            if group_type == SOUND_EFFECT_GROUP_TYPE:
                yield sound_effect_table_offset
            group_offset = following_group_offset

    def _read_sound_effect_table(self, table_offset: int) -> dict[int, int]:
        sound_effect_count = struct.unpack_from(">H", self._project_bytes, table_offset)[0]
        first_entry_offset = table_offset + SOUND_EFFECT_COUNT_SIZE
        return dict(
            struct.unpack_from(
                SOUND_EFFECT_ENTRY_FORMAT,
                self._project_bytes,
                first_entry_offset + entry_index * SOUND_EFFECT_ENTRY_SIZE,
            )
            for entry_index in range(sound_effect_count)
        )

    def sample_id_by_sound_effect_id(self, macro_pool: SoundMacroPool) -> dict[int, int]:
        return {
            sound_effect_id: macro_pool.sample_id_by_macro_id[macro_id]
            for sound_effect_id, macro_id in self.macro_id_by_sound_effect_id.items()
            if macro_id in macro_pool.sample_id_by_macro_id
        }
