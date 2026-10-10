from pathlib import Path

from smbaudio.audio_files import ReplacementAudioFile
from smbaudio.disc.modified_disc_image import DiscImageLayout, ModifiedDiscImage
from smbaudio.errors import InvalidReplacementsError
from smbaudio.game_disc import GameDisc
from smbaudio.music.music_replacement import MusicReplacementPlan
from smbaudio.replacement_config import ReplacementConfig
from smbaudio.sound_effects.sound_bank_rebuilding import AudioMemoryUse, PlannedSoundBank


class DiscImagePatch:
    def __init__(self, game_disc: GameDisc, replacements: dict[str, ReplacementAudioFile]):
        self._disc_image = game_disc.disc_image
        self._music_plan = MusicReplacementPlan(game_disc.music_catalog, replacements)
        self._planned_banks = [PlannedSoundBank.plan(sound_bank, replacements) for sound_bank in game_disc.sound_banks]

    @property
    def _changed_banks(self) -> list[PlannedSoundBank]:
        return [planned_bank for planned_bank in self._planned_banks if planned_bank.is_changed]

    @property
    def _problems(self) -> list[str]:
        audio_memory_use = AudioMemoryUse(
            {planned_bank.sound_bank.name: planned_bank.sample_data_size for planned_bank in self._planned_banks}
        )
        return [
            *self._music_plan.empty_replacement_problems,
            *(problem for planned_bank in self._changed_banks for problem in planned_bank.empty_replacement_problems),
            *audio_memory_use.problems,
        ]

    @property
    def _replaced_size_by_disc_path(self) -> dict[str, int]:
        return {
            **self._music_plan.encoded_size_by_disc_path,
            **{
                planned_bank.sound_bank.sample_data_disc_path: planned_bank.sample_data_size
                for planned_bank in self._changed_banks
            },
        }

    def validate(self) -> None:
        problems = self._problems
        if problems:
            raise InvalidReplacementsError(problems)
        DiscImageLayout.of_disc_image(self._disc_image).with_replaced_sizes(self._replaced_size_by_disc_path)

    def write_to(self, output_image_path: Path) -> None:
        replaced_files = {**self._encoded_music(), **self._rebuilt_sound_bank_files()}
        executable_patches = self._music_plan.executable_patches
        for executable_patch in executable_patches:
            print(executable_patch)
        ModifiedDiscImage(self._disc_image, replaced_files, executable_patches).write_to(output_image_path)
        print(f"Wrote {output_image_path}")

    def _encoded_music(self) -> dict[str, bytes]:
        encoded_files = {}
        for planned_file in self._music_plan.planned_files:
            encoded_files[planned_file.music_file.disc_path] = self._music_plan.encoded(planned_file)
            print(planned_file.size_change)
        return encoded_files

    def _rebuilt_sound_bank_files(self) -> dict[str, bytes]:
        rebuilt_files = {}
        for planned_bank in self._changed_banks:
            for planned_sample in planned_bank.planned_samples:
                print(planned_sample.size_change)
            rebuilt = planned_bank.rebuilt()
            rebuilt_files[planned_bank.sound_bank.sample_directory_disc_path] = rebuilt.sample_directory_bytes
            rebuilt_files[planned_bank.sound_bank.sample_data_disc_path] = rebuilt.sample_data_bytes
            print(planned_bank.sample_data_size_change)
        return rebuilt_files


def patch_disc_image(config_path: Path, input_image_path: Path, output_image_path: Path) -> None:
    game_disc = GameDisc.open_verified(input_image_path)
    replacements = ReplacementConfig(config_path).replacements_checked_against(game_disc.replaceable_names)
    patch = DiscImagePatch(game_disc, replacements)
    patch.validate()
    patch.write_to(output_image_path)
