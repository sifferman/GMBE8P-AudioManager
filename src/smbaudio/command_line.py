import argparse
import sys
from pathlib import Path

from smbaudio.errors import AudioManagerError
from smbaudio.game_disc import GameDisc
from smbaudio.music.music_catalog import MusicDumper
from smbaudio.patch_command import patch_disc_image
from smbaudio.reaper_project import write_replacement_config


def dump_all_audio(image_path: Path, output_directory: Path) -> None:
    game_disc = GameDisc.open_verified(image_path)
    MusicDumper(game_disc.disc_image, output_directory).dump_files(list(game_disc.music_files.values()))
    for sound_bank in game_disc.sound_banks:
        sound_bank.dump(output_directory)
    print(
        f"Dumped {len(game_disc.music_files)} music files and {len(game_disc.sound_effect_names)} sound effects "
        f"to {output_directory}"
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smbaudio",
        description="Replace music and sound effects in Super Monkey Ball (GameCube, NTSC-U, GMBE8P).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dump = commands.add_parser("dump", help="decode every music track and sound effect in the ISO to WAV")
    dump.add_argument("iso", type=Path, help="retail Super Monkey Ball (USA) ISO")
    dump.add_argument("-o", "--output", type=Path, required=True, help="folder to write the WAV files to")

    patch = commands.add_parser("patch", help="write a new ISO with the replacements listed in a JSON file")
    patch.add_argument("--json", type=Path, required=True, help="JSON file mapping original filenames to new audio")
    patch.add_argument("-i", "--input", type=Path, required=True, help="retail ISO (never modified)")
    patch.add_argument("-o", "--output", type=Path, required=True, help="ISO to write")

    reaper2json = commands.add_parser(
        "reaper2json", help="write a JSON file that replaces every original clip in a Reaper project with its render"
    )
    reaper2json.add_argument("-i", "--input", type=Path, required=True, help="Reaper project (.rpp)")
    reaper2json.add_argument("--render", type=Path, required=True, help="audio file the entire project is rendered to")
    reaper2json.add_argument("-o", "--output", type=Path, required=True, help="JSON file to write")
    return parser


def _run_command(arguments: argparse.Namespace) -> None:
    if arguments.command == "dump":
        dump_all_audio(arguments.iso, arguments.output)
    elif arguments.command == "patch":
        patch_disc_image(arguments.json, arguments.input, arguments.output)
    else:
        write_replacement_config(arguments.input, arguments.render, arguments.output)


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    try:
        _run_command(arguments)
    except AudioManagerError as error:
        sys.exit("\n".join(f"error: {problem}" for problem in str(error).splitlines()))
