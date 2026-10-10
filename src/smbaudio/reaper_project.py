import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import rpp

from smbaudio.errors import AudioManagerError, InvalidReplacementsError
from smbaudio.replacement_config import REPLACEMENT_FILE_FIELD, TIME_START_FIELD, TIME_STOP_FIELD

ORIGINAL_NAME_AT_END_OF_PATH = re.compile(r"(?:^|/)((?:music|sfx/[^/]+)/[^/]+\.wav)$")
RENDERS_ENTIRE_PROJECT = "1"
WRITTEN_TIME_DECIMALS = 6


@dataclass(frozen=True)
class ReaperItem:
    source_path: str
    position: float
    length: float

    @property
    def end_position(self) -> float:
        return self.position + self.length

    @property
    def original_name(self) -> str | None:
        found = ORIGINAL_NAME_AT_END_OF_PATH.search(self.source_path.replace("\\", "/"))
        return None if found is None else found.group(1)


class ReaperProject:
    def __init__(self, project_path: Path):
        self._project_path = Path(project_path)
        if not self._project_path.is_file():
            raise AudioManagerError(f"Reaper project does not exist: {self._project_path}")
        self._project = rpp.loads(self._project_path.read_text(encoding="utf-8"))

    @property
    def items_holding_original_audio(self) -> list[ReaperItem]:
        items = [
            ReaperItem(source_file[1], float(item.find("POSITION")[1]), float(item.find("LENGTH")[1]))
            for item in self._project.findall(".//ITEM")
            for source_file in item.findall(".//SOURCE/FILE")
        ]
        return sorted(
            (item for item in items if item.original_name is not None),
            key=lambda item: (item.position, item.source_path),
        )

    @property
    def problems(self) -> list[str]:
        original_names = [item.original_name for item in self.items_holding_original_audio]
        repeated_names = sorted({name for name in original_names if original_names.count(name) > 1})
        return [
            *(["the project play rate must be 1"] if float(self._setting("PLAYRATE")[0]) != 1 else []),
            *(
                ["the render bounds must be the entire project, so that the render starts at time 0"]
                if self._setting("RENDER_RANGE")[0] != RENDERS_ENTIRE_PROJECT
                else []
            ),
            *(["the project holds no dumped original audio"] if not original_names else []),
            *(f"{name} is in the project more than once" for name in repeated_names),
        ]

    def _setting(self, setting_name: str) -> list[str]:
        return self._project.find(setting_name)[1:]


def _path_written_in_config(file_path: Path, config_path: Path) -> str:
    try:
        return Path(os.path.relpath(file_path, config_path.parent)).as_posix()
    except ValueError:
        return file_path.resolve().as_posix()


def write_replacement_config(project_path: Path, rendered_file_path: Path, config_path: Path) -> None:
    project = ReaperProject(project_path)
    if project.problems:
        raise InvalidReplacementsError(project.problems)
    replacement_file = _path_written_in_config(Path(rendered_file_path), Path(config_path))
    config = {
        item.original_name: {
            REPLACEMENT_FILE_FIELD: replacement_file,
            TIME_START_FIELD: round(item.position, WRITTEN_TIME_DECIMALS),
            TIME_STOP_FIELD: round(item.end_position, WRITTEN_TIME_DECIMALS),
        }
        for item in project.items_holding_original_audio
    }
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {config_path}: {len(config)} original files replaced by time slots of {rendered_file_path}")
    if not Path(rendered_file_path).is_file():
        print(f"{rendered_file_path} does not exist yet; render the project in Reaper before patching")
