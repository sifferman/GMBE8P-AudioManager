import json
from numbers import Real
from pathlib import Path

from smbaudio.audio_files import ReplacementAudioFile
from smbaudio.errors import AudioManagerError, InvalidReplacementsError

REPLACEMENT_FILE_FIELD = "replacement_file"
TIME_START_FIELD = "time_start"
TIME_STOP_FIELD = "time_stop"
ENTRY_FIELDS = {REPLACEMENT_FILE_FIELD, TIME_START_FIELD, TIME_STOP_FIELD}


def _normalized_original_name(original_name: str) -> str:
    return original_name.replace("\\", "/").lstrip("./")


def _is_time_in_seconds(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and value >= 0


def _is_valid_entry(entry: object) -> bool:
    if isinstance(entry, str):
        return True
    if not isinstance(entry, dict) or not set(entry) <= ENTRY_FIELDS:
        return False
    time_start = entry.get(TIME_START_FIELD, 0)
    time_stop = entry.get(TIME_STOP_FIELD)
    return (
        isinstance(entry.get(REPLACEMENT_FILE_FIELD), str)
        and _is_time_in_seconds(time_start)
        and (time_stop is None or (_is_time_in_seconds(time_stop) and time_stop > time_start))
    )


class ReplacementConfig:
    def __init__(self, config_path: Path):
        self._config_path = Path(config_path)
        self.replacements = {
            _normalized_original_name(original_name): self._replacement(entry)
            for original_name, entry in self._read_entries().items()
        }

    def replacements_checked_against(self, replaceable_names: set[str]) -> dict[str, ReplacementAudioFile]:
        problems = self._unknown_name_problems(replaceable_names) + self._unreadable_file_problems()
        if problems:
            raise InvalidReplacementsError(problems)
        return self.replacements

    def _replacement(self, entry: str | dict) -> ReplacementAudioFile:
        fields = {REPLACEMENT_FILE_FIELD: entry} if isinstance(entry, str) else entry
        written_path = fields[REPLACEMENT_FILE_FIELD]
        return ReplacementAudioFile(
            written_path,
            self._config_path.parent / written_path,
            fields.get(TIME_START_FIELD, 0),
            fields.get(TIME_STOP_FIELD),
        )

    def _read_entries(self) -> dict[str, str | dict]:
        if not self._config_path.is_file():
            raise AudioManagerError(f"JSON file does not exist: {self._config_path}")
        try:
            config = json.loads(self._config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as json_error:
            raise AudioManagerError(f"{self._config_path} is not valid JSON: {json_error}") from json_error
        if not isinstance(config, dict):
            raise AudioManagerError(f"{self._config_path} must be a dictionary keyed by original filename")
        invalid_names = [original_name for original_name, entry in config.items() if not _is_valid_entry(entry)]
        if invalid_names:
            raise InvalidReplacementsError(
                [
                    f"{original_name} must map to a replacement file path, or to a dictionary with "
                    f"{REPLACEMENT_FILE_FIELD} and optionally {TIME_START_FIELD} and {TIME_STOP_FIELD} in seconds"
                    for original_name in invalid_names
                ]
            )
        return config

    def _unknown_name_problems(self, replaceable_names: set[str]) -> list[str]:
        return [
            f"unknown original filename: {original_name}"
            for original_name in self.replacements
            if original_name not in replaceable_names
        ]

    def _unreadable_file_problems(self) -> list[str]:
        reasons = [replacement.reason_it_cannot_be_read for replacement in self.replacements.values()]
        return list(dict.fromkeys(reason for reason in reasons if reason is not None))
