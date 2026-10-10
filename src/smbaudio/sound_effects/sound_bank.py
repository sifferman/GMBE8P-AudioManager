import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from smbaudio.audio_files import write_wav_file
from smbaudio.disc.dol_executable import DolExecutable
from smbaudio.disc.gamecube_disc_image import GameCubeDiscImage
from smbaudio.sound_effects.dsp_adpcm import EncodedDspAdpcm, decode_dsp_adpcm_streams
from smbaudio.sound_effects.musyx_files import DirectorySample, SampleDirectory, SoundEffectProject, SoundMacroPool

NAME_TABLE_MEMORY_ADDRESS = 0x8011057C
NAME_TABLE_ENTRY_FORMAT = ">iI8x"
NAME_TABLE_ENTRY_SIZE = struct.calcsize(NAME_TABLE_ENTRY_FORMAT)
NAME_TABLE_ENTRY_COUNT = 0x425
SECTION_NAME_PREFIX = "GRP"
SOUND_BANK_DISC_DIRECTORY = "test/snd/mkb/"
DUMPED_SOUND_EFFECT_DIRECTORY = "sfx/"


@dataclass(frozen=True)
class NameTableEntry:
    identifier: int
    name: str

    @property
    def starts_section(self) -> bool:
        return self.name.startswith(SECTION_NAME_PREFIX)

    @property
    def has_usable_identifier(self) -> bool:
        return self.identifier >= 0


class SoundEffectNameTable:
    def __init__(self, executable: DolExecutable):
        self._executable = executable
        self.first_name_by_sound_effect_id: dict[int, str] = {}
        section_holds_sound_effects = False
        for entry_index in range(NAME_TABLE_ENTRY_COUNT):
            entry = self._read_entry(entry_index)
            if entry.starts_section:
                section_holds_sound_effects = entry.has_usable_identifier
            elif section_holds_sound_effects and entry.has_usable_identifier:
                self.first_name_by_sound_effect_id.setdefault(entry.identifier, entry.name)

    def _read_entry(self, entry_index: int) -> NameTableEntry:
        identifier, name_address = struct.unpack(
            NAME_TABLE_ENTRY_FORMAT,
            self._executable.read_bytes_at_memory_address(
                NAME_TABLE_MEMORY_ADDRESS + entry_index * NAME_TABLE_ENTRY_SIZE, NAME_TABLE_ENTRY_SIZE
            ),
        )
        return NameTableEntry(identifier, self._executable.read_text_at_memory_address(name_address))


@dataclass(frozen=True)
class SoundBank:
    name: str
    sample_directory: SampleDirectory
    sample_data: bytes
    dumped_name_by_sample_id: dict[int, str]

    @staticmethod
    def disc_path(bank_name: str, extension: str) -> str:
        return f"{SOUND_BANK_DISC_DIRECTORY}{bank_name}.{extension}"

    @property
    def sample_directory_disc_path(self) -> str:
        return self.disc_path(self.name, "sdir")

    @property
    def sample_data_disc_path(self) -> str:
        return self.disc_path(self.name, "samp")

    @property
    def samples(self) -> tuple[DirectorySample, ...]:
        return self.sample_directory.samples

    @property
    def samples_in_sample_data_order(self) -> list[DirectorySample]:
        return sorted(self.samples, key=lambda sample: sample.data_offset)

    @property
    def dumped_names(self) -> set[str]:
        return set(self.dumped_name_by_sample_id.values())

    def dumped_name(self, sample: DirectorySample) -> str:
        return self.dumped_name_by_sample_id[sample.sample_id]

    def encoded_bytes(self, sample: DirectorySample) -> bytes:
        return self.sample_data[sample.data_offset :][: sample.encoded_size]

    def encoded_sample(self, sample: DirectorySample) -> EncodedDspAdpcm:
        return EncodedDspAdpcm(self.encoded_bytes(sample), sample.coding.coefficients, sample.sample_count)

    def dump(self, output_directory: Path) -> None:
        decoded_samples = decode_dsp_adpcm_streams([self.encoded_sample(sample) for sample in self.samples])
        for sample, decoded in zip(self.samples, decoded_samples):
            write_wav_file(output_directory / self.dumped_name(sample), decoded, sample.sample_rate)
            print(self.dumped_name(sample))


class SoundBankReader:
    def __init__(self, disc_image: GameCubeDiscImage, executable: DolExecutable):
        self._disc_image = disc_image
        self._name_by_sound_effect_id = SoundEffectNameTable(executable).first_name_by_sound_effect_id

    def read_all_sound_banks(self) -> list[SoundBank]:
        return [self.read_sound_bank(bank_name) for bank_name in self._bank_names()]

    def read_sound_bank(self, bank_name: str) -> SoundBank:
        sample_directory = SampleDirectory.parse(self._read_bank_file(bank_name, "sdir"))
        name_by_sample_id = self._first_name_by_sample_id(bank_name)
        return SoundBank(
            bank_name,
            sample_directory,
            self._read_bank_file(bank_name, "samp"),
            {
                sample.sample_id: self._dumped_name(bank_name, sample, name_by_sample_id)
                for sample in sample_directory.samples
            },
        )

    def _bank_names(self) -> list[str]:
        return sorted(
            PurePosixPath(disc_file.path).stem
            for disc_file in self._disc_image.files_in_directory(SOUND_BANK_DISC_DIRECTORY)
            if disc_file.path.endswith(".samp")
        )

    def _read_bank_file(self, bank_name: str, extension: str) -> bytes:
        return self._disc_image.read_file(SoundBank.disc_path(bank_name, extension))

    def _first_name_by_sample_id(self, bank_name: str) -> dict[int, str]:
        sound_effect_project = SoundEffectProject(self._read_bank_file(bank_name, "proj"))
        sound_macro_pool = SoundMacroPool(self._read_bank_file(bank_name, "pool"))
        sample_ids = sound_effect_project.sample_id_by_sound_effect_id(sound_macro_pool)
        names: dict[int, str] = {}
        for sound_effect_id, sample_id in sorted(sample_ids.items()):
            if sound_effect_id in self._name_by_sound_effect_id:
                names.setdefault(sample_id, self._name_by_sound_effect_id[sound_effect_id])
        return names

    @staticmethod
    def _dumped_name(bank_name: str, sample: DirectorySample, name_by_sample_id: dict[int, str]) -> str:
        sample_name = name_by_sample_id.get(sample.sample_id, f"unnamed_sample_{sample.sample_id}")
        return f"{DUMPED_SOUND_EFFECT_DIRECTORY}{bank_name}/{sample_name}.wav"
