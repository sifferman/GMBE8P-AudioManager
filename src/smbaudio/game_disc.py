import hashlib
from dataclasses import dataclass
from pathlib import Path

from smbaudio.disc.gamecube_disc_image import GameCubeDiscImage
from smbaudio.errors import AudioManagerError
from smbaudio.music.music_catalog import MusicCatalog, MusicFile
from smbaudio.sound_effects.sound_bank import SoundBank, SoundBankReader

RETAIL_DISC_IMAGE_SHA1 = "2cca02879db51ca2850949c7b2349cf470607cf5"
HASHED_CHUNK_SIZE = 1 << 24


def disc_image_sha1(image_path: Path) -> str:
    sha1 = hashlib.sha1()
    with open(image_path, "rb") as image:
        while chunk := image.read(HASHED_CHUNK_SIZE):
            sha1.update(chunk)
    return sha1.hexdigest()


def verify_retail_disc_image(image_path: Path) -> None:
    if not Path(image_path).is_file():
        raise AudioManagerError(f"ISO does not exist: {image_path}")
    actual_sha1 = disc_image_sha1(image_path)
    if actual_sha1 != RETAIL_DISC_IMAGE_SHA1:
        raise AudioManagerError(
            f"{image_path} is not the retail Super Monkey Ball (USA, GMBE8P) ISO.\n"
            f"  expected SHA-1 {RETAIL_DISC_IMAGE_SHA1}\n"
            f"  actual SHA-1   {actual_sha1}"
        )


@dataclass(frozen=True)
class GameDisc:
    disc_image: GameCubeDiscImage
    music_catalog: MusicCatalog
    sound_banks: list[SoundBank]

    @classmethod
    def open_verified(cls, image_path: Path) -> "GameDisc":
        verify_retail_disc_image(image_path)
        disc_image = GameCubeDiscImage(image_path)
        executable = disc_image.read_executable()
        return cls(
            disc_image,
            MusicCatalog(disc_image, executable),
            SoundBankReader(disc_image, executable).read_all_sound_banks(),
        )

    @property
    def music_files(self) -> dict[str, MusicFile]:
        return self.music_catalog.music_files

    @property
    def sound_effect_names(self) -> set[str]:
        return {dumped_name for sound_bank in self.sound_banks for dumped_name in sound_bank.dumped_names}

    @property
    def replaceable_names(self) -> set[str]:
        return set(self.music_files) | self.sound_effect_names
