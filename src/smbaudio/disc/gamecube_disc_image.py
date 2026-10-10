import struct
from dataclasses import dataclass
from pathlib import Path

from smbaudio.disc.dol_executable import HEADER_SIZE, DolExecutable

TABLE_ENTRY_SIZE = 12
DIRECTORY_FLAG = 1
EXECUTABLE_OFFSET_FIELD = 0x420
FILE_SYSTEM_TABLE_OFFSET_FIELD = 0x424
FILE_SYSTEM_TABLE_SIZE_FIELD = 0x428
GAME_ID_SIZE = 6


@dataclass(frozen=True)
class DiscFile:
    path: str
    table_entry_index: int
    offset: int
    size: int

    @property
    def end_offset(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class TableEntry:
    flag: int
    name_offset: int
    offset_on_disc: int
    size_or_following_entry_index: int

    @property
    def is_directory(self) -> bool:
        return self.flag == DIRECTORY_FLAG


@dataclass(frozen=True)
class OpenDirectory:
    path: str
    following_entry_index: int


class FileSystemTable:
    def __init__(self, table_bytes: bytes):
        self.table_bytes = table_bytes
        self._entry_count = self._read_entry(0).size_or_following_entry_index
        self.files = self._read_files()

    def with_relocated_files(self, relocated_files: list[DiscFile]) -> "FileSystemTable":
        relocated_table = bytearray(self.table_bytes)
        for disc_file in relocated_files:
            struct.pack_into(
                ">II",
                relocated_table,
                disc_file.table_entry_index * TABLE_ENTRY_SIZE + 4,
                disc_file.offset,
                disc_file.size,
            )
        return FileSystemTable(bytes(relocated_table))

    def _read_entry(self, entry_index: int) -> TableEntry:
        flag_and_name_offset, second_field, third_field = struct.unpack_from(
            ">III", self.table_bytes, entry_index * TABLE_ENTRY_SIZE
        )
        return TableEntry(flag_and_name_offset >> 24, flag_and_name_offset & 0xFFFFFF, second_field, third_field)

    def _read_name(self, name_offset: int) -> str:
        name_start = self._entry_count * TABLE_ENTRY_SIZE + name_offset
        return self.table_bytes[name_start : self.table_bytes.index(b"\0", name_start)].decode("ascii")

    def _read_files(self) -> list[DiscFile]:
        open_directories: list[OpenDirectory] = []
        disc_files = []
        for entry_index in range(1, self._entry_count):
            while open_directories and entry_index >= open_directories[-1].following_entry_index:
                open_directories.pop()
            entry = self._read_entry(entry_index)
            parent_path = open_directories[-1].path if open_directories else ""
            path = parent_path + self._read_name(entry.name_offset)
            if entry.is_directory:
                open_directories.append(OpenDirectory(path + "/", entry.size_or_following_entry_index))
            else:
                disc_files.append(
                    DiscFile(path, entry_index, entry.offset_on_disc, entry.size_or_following_entry_index)
                )
        return disc_files


class GameCubeDiscImage:
    def __init__(self, image_path: Path):
        self.image_path = Path(image_path)
        self.image_size = self.image_path.stat().st_size
        self.game_id = self.read_bytes(0, GAME_ID_SIZE).decode("ascii", errors="replace")
        self.executable_offset = self._read_header_field(EXECUTABLE_OFFSET_FIELD)
        self.file_system_table_offset = self._read_header_field(FILE_SYSTEM_TABLE_OFFSET_FIELD)
        self.file_system_table = FileSystemTable(
            self.read_bytes(self.file_system_table_offset, self._read_header_field(FILE_SYSTEM_TABLE_SIZE_FIELD))
        )
        self._files_by_path = {disc_file.path: disc_file for disc_file in self.files}

    @property
    def files(self) -> list[DiscFile]:
        return self.file_system_table.files

    def _read_header_field(self, field_offset: int) -> int:
        return struct.unpack(">I", self.read_bytes(field_offset, 4))[0]

    def read_bytes(self, offset: int, size: int) -> bytes:
        with open(self.image_path, "rb") as image:
            image.seek(offset)
            return image.read(size)

    def read_executable(self) -> DolExecutable:
        header_bytes = self.read_bytes(self.executable_offset, HEADER_SIZE)
        executable_size = DolExecutable.size_stated_in_header(header_bytes)
        return DolExecutable(self.read_bytes(self.executable_offset, executable_size))

    def find_file(self, path: str) -> DiscFile:
        return self._files_by_path[path]

    def read_file(self, path: str) -> bytes:
        return self.read_disc_file(self.find_file(path))

    def read_disc_file(self, disc_file: DiscFile) -> bytes:
        return self.read_bytes(disc_file.offset, disc_file.size)

    def read_start_of_disc_file(self, disc_file: DiscFile, largest_size: int) -> bytes:
        return self.read_bytes(disc_file.offset, min(largest_size, disc_file.size))

    def files_in_directory(self, directory_path: str) -> list[DiscFile]:
        return [disc_file for disc_file in self.files if disc_file.path.startswith(directory_path)]
