import struct
from dataclasses import replace

from smbaudio.disc.gamecube_disc_image import FileSystemTable

NAMES = b"snd\0a.adp\0b.bin\0"
TABLE = (
    struct.pack(">III", 1 << 24, 0, 4)
    + struct.pack(">III", 1 << 24, 0, 3)
    + struct.pack(">III", 4, 0x8000, 0x10000)
    + struct.pack(">III", 10, 0x20000, 0x123)
    + NAMES
)


def test_files_inside_directories_get_full_paths():
    disc_files = FileSystemTable(TABLE).files
    assert [(disc_file.path, disc_file.offset, disc_file.size) for disc_file in disc_files] == [
        ("snd/a.adp", 0x8000, 0x10000),
        ("b.bin", 0x20000, 0x123),
    ]


def test_relocated_files_are_written_back_to_the_table():
    table = FileSystemTable(TABLE)
    first_file, second_file = table.files
    moved_file = replace(first_file, offset=0x30000, size=0x18000)
    assert table.with_relocated_files([moved_file]).files == [moved_file, second_file]
