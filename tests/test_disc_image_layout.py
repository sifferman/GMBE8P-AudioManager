from itertools import pairwise

import pytest

from smbaudio.disc.gamecube_disc_image import DiscFile
from smbaudio.disc.modified_disc_image import DiscImageLayout
from smbaudio.errors import AudioManagerError

FILES = [
    DiscFile("first.adp", 1, 0x10000, 0x8000),
    DiscFile("second.adp", 2, 0x18000, 0x8000),
    DiscFile("third.bin", 3, 0x40000, 0x100),
]
IMAGE_SIZE = 0x60000
LAYOUT = DiscImageLayout(FILES, IMAGE_SIZE)


def offsets_by_path(laid_out_files: list[DiscFile]) -> dict[str, int]:
    return {disc_file.path: disc_file.offset for disc_file in laid_out_files}


def test_files_keep_their_offsets_when_nothing_grows():
    assert offsets_by_path(LAYOUT.with_replaced_sizes({"first.adp": 0x4000})) == {
        "first.adp": 0x10000,
        "second.adp": 0x18000,
        "third.bin": 0x40000,
    }


def test_grown_file_moves_to_aligned_free_space_and_others_stay():
    offsets = offsets_by_path(LAYOUT.with_replaced_sizes({"first.adp": 0x10000}))
    assert offsets == {"first.adp": 0x20000, "second.adp": 0x18000, "third.bin": 0x40000}


def test_laid_out_files_never_overlap():
    laid_out = sorted(
        LAYOUT.with_replaced_sizes({"first.adp": 0x18000, "second.adp": 0x10000}),
        key=lambda disc_file: disc_file.offset,
    )
    for earlier, later in pairwise(laid_out):
        assert earlier.end_offset <= later.offset
    assert laid_out[-1].end_offset <= IMAGE_SIZE


def test_replacements_that_do_not_fit_are_rejected():
    with pytest.raises(AudioManagerError):
        LAYOUT.with_replaced_sizes({"first.adp": 0x50000})
