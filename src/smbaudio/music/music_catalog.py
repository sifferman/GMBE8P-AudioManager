import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import numpy

from smbaudio.audio_files import write_wav_file
from smbaudio.disc.dol_executable import DolExecutable
from smbaudio.disc.gamecube_disc_image import DiscFile, GameCubeDiscImage
from smbaudio.music.dtk_adpcm import FRAME_SIZE, SAMPLE_RATE, decode_dtk_adpcm, frame_count_for_samples

TRACK_TABLE_MEMORY_ADDRESS = 0x801B2A5C
TRACK_TABLE_ENTRY_FORMAT = ">IIIb3x"
TRACK_TABLE_ENTRY_SIZE = struct.calcsize(TRACK_TABLE_ENTRY_FORMAT)
SURROUND_TRACK_COUNT = 0x4A
TRACK_COUNT = SURROUND_TRACK_COUNT * 2
PLAYED_SAMPLE_COUNT_FIELD_OFFSET = 8
STOP_AFTER_PLAYING = -1
MUSIC_DISC_DIRECTORY = "test/snd/adp/"
DUMPED_MUSIC_DIRECTORY = "music/"


@dataclass(frozen=True)
class MusicTrack:
    table_index: int
    name: str
    played_sample_count: int
    following_table_index: int

    @property
    def is_mono_version(self) -> bool:
        return self.table_index >= SURROUND_TRACK_COUNT

    @property
    def stops_after_playing(self) -> bool:
        return self.following_table_index == STOP_AFTER_PLAYING

    @property
    def table_index_of_other_version(self) -> int:
        return (self.table_index + SURROUND_TRACK_COUNT) % TRACK_COUNT

    @property
    def table_index_holding_played_sample_count(self) -> int:
        return self.table_index % SURROUND_TRACK_COUNT

    @property
    def table_index_played_afterwards(self) -> int:
        return self.following_table_index + (SURROUND_TRACK_COUNT if self.is_mono_version else 0)


class MusicTrackTable:
    def __init__(self, executable: DolExecutable):
        self._executable = executable
        stored_tracks = [self._read_track(table_index) for table_index in range(TRACK_COUNT)]
        self.tracks = [
            replace(
                track,
                played_sample_count=stored_tracks[track.table_index_holding_played_sample_count].played_sample_count,
            )
            for track in stored_tracks
        ]

    def other_version_of(self, track: MusicTrack) -> MusicTrack:
        return self.tracks[track.table_index_of_other_version]

    def track_played_after(self, track: MusicTrack) -> MusicTrack | None:
        return None if track.stops_after_playing else self.tracks[track.table_index_played_afterwards]

    def tracks_sharing_file_with(self, track: MusicTrack) -> list[MusicTrack]:
        return [other_track for other_track in self.tracks if other_track.name == track.name]

    def played_sample_count_file_offset(self, table_index: int) -> int:
        return self._executable.file_offset_of_memory_address(
            self._entry_memory_address(table_index) + PLAYED_SAMPLE_COUNT_FIELD_OFFSET
        )

    def _entry_memory_address(self, table_index: int) -> int:
        return TRACK_TABLE_MEMORY_ADDRESS + table_index * TRACK_TABLE_ENTRY_SIZE

    def _read_track(self, table_index: int) -> MusicTrack:
        name_address, _, played_sample_count, following_table_index = struct.unpack(
            TRACK_TABLE_ENTRY_FORMAT,
            self._executable.read_bytes_at_memory_address(
                self._entry_memory_address(table_index), TRACK_TABLE_ENTRY_SIZE
            ),
        )
        return MusicTrack(
            table_index,
            self._executable.read_text_at_memory_address(name_address),
            played_sample_count,
            following_table_index,
        )


@dataclass(frozen=True)
class MusicFile:
    dumped_name: str
    disc_file: DiscFile
    played_sample_count: int
    is_mono_version: bool
    dumped_name_of_other_version: str
    dumped_name_played_afterwards: str | None
    table_indices_holding_played_sample_count: tuple[int, ...]

    @property
    def disc_path(self) -> str:
        return self.disc_file.path

    @property
    def encoded_name(self) -> str:
        return str(PurePosixPath(self.dumped_name).with_suffix(".adp"))

    def decoded_played_samples(self, disc_image: GameCubeDiscImage) -> numpy.ndarray:
        return decode_dtk_adpcm(disc_image.read_disc_file(self.disc_file))[: self.played_sample_count]

    def decoded_first_samples(self, disc_image: GameCubeDiscImage, sample_count: int) -> numpy.ndarray:
        encoded_size = frame_count_for_samples(sample_count) * FRAME_SIZE
        return decode_dtk_adpcm(disc_image.read_start_of_disc_file(self.disc_file, encoded_size))[:sample_count]


class MusicCatalog:
    def __init__(self, disc_image: GameCubeDiscImage, executable: DolExecutable):
        self.disc_image = disc_image
        self.track_table = MusicTrackTable(executable)
        self._disc_file_by_lowercase_track_name = {
            PurePosixPath(disc_file.path).stem.lower(): disc_file
            for disc_file in disc_image.files_in_directory(MUSIC_DISC_DIRECTORY)
        }
        self.music_files: dict[str, MusicFile] = {}
        for track in self.track_table.tracks:
            self.music_files.setdefault(self._dumped_name(track), self._music_file(track))

    def other_version_of(self, music_file: MusicFile) -> MusicFile:
        return self.music_files[music_file.dumped_name_of_other_version]

    def file_played_after(self, music_file: MusicFile) -> MusicFile | None:
        if music_file.dumped_name_played_afterwards is None:
            return None
        return self.music_files[music_file.dumped_name_played_afterwards]

    def _disc_file(self, track: MusicTrack) -> DiscFile:
        return self._disc_file_by_lowercase_track_name[track.name.lower()]

    def _dumped_name(self, track: MusicTrack) -> str:
        return DUMPED_MUSIC_DIRECTORY + PurePosixPath(self._disc_file(track).path).stem + ".wav"

    def _dumped_name_played_afterwards(self, track: MusicTrack) -> str | None:
        track_played_afterwards = self.track_table.track_played_after(track)
        return None if track_played_afterwards is None else self._dumped_name(track_played_afterwards)

    def _music_file(self, track: MusicTrack) -> MusicFile:
        return MusicFile(
            self._dumped_name(track),
            self._disc_file(track),
            track.played_sample_count,
            track.is_mono_version,
            self._dumped_name(self.track_table.other_version_of(track)),
            self._dumped_name_played_afterwards(track),
            tuple(
                track_sharing_file.table_index_holding_played_sample_count
                for track_sharing_file in self.track_table.tracks_sharing_file_with(track)
            ),
        )


@dataclass(frozen=True)
class MusicDumper:
    disc_image: GameCubeDiscImage
    output_directory: Path

    def dump_file(self, music_file: MusicFile) -> str:
        write_wav_file(
            self.output_directory / music_file.dumped_name,
            music_file.decoded_played_samples(self.disc_image),
            SAMPLE_RATE,
        )
        return music_file.dumped_name

    def dump_files(self, music_files: list[MusicFile]) -> None:
        with ProcessPoolExecutor() as worker_processes:
            for dumped_name in worker_processes.map(self.dump_file, music_files):
                print(dumped_name)
