import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

from smbaudio.disc.dol_executable import ExecutablePatch
from smbaudio.disc.gamecube_disc_image import DiscFile, GameCubeDiscImage
from smbaudio.errors import AudioManagerError

RELOCATED_FILE_ALIGNMENT = 0x8000


@dataclass(frozen=True)
class FreeRegion:
    start_offset: int
    end_offset: int

    @property
    def is_empty(self) -> bool:
        return self.end_offset <= self.start_offset

    @property
    def aligned_start_offset(self) -> int:
        return self.start_offset + -self.start_offset % RELOCATED_FILE_ALIGNMENT

    def has_room_for(self, size: int) -> bool:
        return self.aligned_start_offset + size <= self.end_offset

    def after_placing(self, size: int) -> "FreeRegion":
        return FreeRegion(self.aligned_start_offset + size, self.end_offset)


class FreeSpace:
    def __init__(self, free_regions: list[FreeRegion]):
        self._free_regions = [region for region in free_regions if not region.is_empty]

    def place(self, disc_file: DiscFile) -> DiscFile:
        for region_index, region in enumerate(self._free_regions):
            if region.has_room_for(disc_file.size):
                self._free_regions[region_index] = region.after_placing(disc_file.size)
                return replace(disc_file, offset=region.aligned_start_offset)
        raise AudioManagerError(
            f"The replacements do not fit on the disc: no room left for {disc_file.path} ({disc_file.size} bytes)"
        )


class DiscImageLayout:
    def __init__(self, files: list[DiscFile], image_size: int):
        self._files_by_offset = sorted(files, key=lambda disc_file: disc_file.offset)
        self._image_size = image_size

    @classmethod
    def of_disc_image(cls, disc_image: GameCubeDiscImage) -> "DiscImageLayout":
        return cls(disc_image.files, disc_image.image_size)

    def with_replaced_sizes(self, replaced_sizes: dict[str, int]) -> list[DiscFile]:
        room_by_path = self._room_until_following_file_by_path()
        resized_files = [
            replace(disc_file, size=replaced_sizes.get(disc_file.path, disc_file.size))
            for disc_file in self._files_by_offset
        ]
        staying_files = [disc_file for disc_file in resized_files if disc_file.size <= room_by_path[disc_file.path]]
        moving_files = [disc_file for disc_file in resized_files if disc_file.size > room_by_path[disc_file.path]]
        free_space = self._free_space_around(staying_files)
        largest_files_first = sorted(moving_files, key=lambda moving_file: -moving_file.size)
        return staying_files + [free_space.place(disc_file) for disc_file in largest_files_first]

    def _room_until_following_file_by_path(self) -> dict[str, int]:
        following_offsets = [disc_file.offset for disc_file in self._files_by_offset[1:]] + [self._image_size]
        return {
            disc_file.path: following_offset - disc_file.offset
            for disc_file, following_offset in zip(self._files_by_offset, following_offsets)
        }

    def _free_space_around(self, staying_files: list[DiscFile]) -> FreeSpace:
        region_start_offsets = [self._files_by_offset[0].offset] + [disc_file.end_offset for disc_file in staying_files]
        region_end_offsets = [disc_file.offset for disc_file in staying_files] + [self._image_size]
        return FreeSpace(
            [
                FreeRegion(start_offset, end_offset)
                for start_offset, end_offset in zip(region_start_offsets, region_end_offsets)
            ]
        )


class ModifiedDiscImage:
    def __init__(
        self,
        original_image: GameCubeDiscImage,
        replaced_files: dict[str, bytes],
        executable_patches: list[ExecutablePatch],
    ):
        self._original_image = original_image
        self._replaced_files = replaced_files
        self._executable_patches = executable_patches
        self._laid_out_files = DiscImageLayout.of_disc_image(original_image).with_replaced_sizes(
            {path: len(contents) for path, contents in replaced_files.items()}
        )

    def write_to(self, output_path: Path) -> None:
        output_path = Path(output_path)
        if output_path.resolve() == self._original_image.image_path.resolve():
            raise AudioManagerError("The output ISO must be a different file from the input ISO")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._original_image.image_path, output_path)
        with open(output_path, "r+b") as output_image:
            self._write_changed_files(output_image)
            self._write_file_system_table(output_image)
            self._write_executable_patches(output_image)

    def _write_changed_files(self, output_image: BinaryIO) -> None:
        for disc_file in self._laid_out_files:
            if self._is_replaced(disc_file) or self._is_relocated(disc_file):
                output_image.seek(disc_file.offset)
                output_image.write(self._contents(disc_file))

    def _write_file_system_table(self, output_image: BinaryIO) -> None:
        relocated_table = self._original_image.file_system_table.with_relocated_files(self._laid_out_files)
        output_image.seek(self._original_image.file_system_table_offset)
        output_image.write(relocated_table.table_bytes)

    def _write_executable_patches(self, output_image: BinaryIO) -> None:
        for patch in self._executable_patches:
            output_image.seek(self._original_image.executable_offset + patch.file_offset)
            output_image.write(patch.patched_bytes)

    def _is_replaced(self, disc_file: DiscFile) -> bool:
        return disc_file.path in self._replaced_files

    def _is_relocated(self, disc_file: DiscFile) -> bool:
        return disc_file.offset != self._original_image.find_file(disc_file.path).offset

    def _contents(self, disc_file: DiscFile) -> bytes:
        if self._is_replaced(disc_file):
            return self._replaced_files[disc_file.path]
        return self._original_image.read_file(disc_file.path)
