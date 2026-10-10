import struct
from dataclasses import dataclass

SECTION_COUNT = 18
SECTION_FILE_OFFSETS_FIELD = 0x00
SECTION_MEMORY_ADDRESSES_FIELD = 0x48
SECTION_SIZES_FIELD = 0x90
HEADER_SIZE = 0x100


@dataclass(frozen=True)
class ExecutableSection:
    file_offset: int
    memory_address: int
    size: int

    @property
    def end_file_offset(self) -> int:
        return self.file_offset + self.size

    def holds_memory_address(self, memory_address: int) -> bool:
        return self.memory_address <= memory_address < self.memory_address + self.size

    def file_offset_of_memory_address(self, memory_address: int) -> int:
        return self.file_offset + memory_address - self.memory_address


def _read_section_fields(header_bytes: bytes, field_offset: int) -> tuple[int, ...]:
    return struct.unpack_from(f">{SECTION_COUNT}I", header_bytes, field_offset)


def _read_sections(header_bytes: bytes) -> list[ExecutableSection]:
    return [
        ExecutableSection(file_offset, memory_address, size)
        for file_offset, memory_address, size in zip(
            _read_section_fields(header_bytes, SECTION_FILE_OFFSETS_FIELD),
            _read_section_fields(header_bytes, SECTION_MEMORY_ADDRESSES_FIELD),
            _read_section_fields(header_bytes, SECTION_SIZES_FIELD),
        )
    ]


class DolExecutable:
    def __init__(self, executable_bytes: bytes):
        self.executable_bytes = executable_bytes
        self._sections = _read_sections(executable_bytes)

    @staticmethod
    def size_stated_in_header(header_bytes: bytes) -> int:
        return max(section.end_file_offset for section in _read_sections(header_bytes))

    def file_offset_of_memory_address(self, memory_address: int) -> int:
        for section in self._sections:
            if section.holds_memory_address(memory_address):
                return section.file_offset_of_memory_address(memory_address)
        raise ValueError(f"Memory address {memory_address:#010x} is not inside the executable")

    def read_bytes_at_memory_address(self, memory_address: int, size: int) -> bytes:
        file_offset = self.file_offset_of_memory_address(memory_address)
        return self.executable_bytes[file_offset : file_offset + size]

    def read_text_at_memory_address(self, memory_address: int) -> str:
        file_offset = self.file_offset_of_memory_address(memory_address)
        text_end = self.executable_bytes.index(b"\0", file_offset)
        return self.executable_bytes[file_offset:text_end].decode("ascii")


@dataclass(frozen=True)
class ExecutablePatch:
    file_offset: int
    patched_bytes: bytes
    description: str

    def __str__(self) -> str:
        return f"main.dol {self.file_offset:#08x}  {self.description}"
