import json
from pathlib import Path

import pytest

from smbaudio.errors import InvalidReplacementsError
from smbaudio.reaper_project import write_replacement_config

ORIGINAL_CLIPS = (
    (1.5, 2, r"..\..\dumped_audio\sfx\comn\SFXready_01_basic_macro.wav"),
    (0, 1.25, "C:/dumps/music/st1_lp.wav"),
    (0, 1.25, r"..\..\dumped_audio\sfx\allse\SND_SED_NAR_READY.wav"),
)
NEW_VOICE_CLIP = (0.1, 3, r"recordings\new voice.wav")


def item_text(position: float, length: float, source_path: str) -> str:
    return f"""    <ITEM
      POSITION {position}
      LENGTH {length}
      <SOURCE WAVE
        FILE "{source_path}"
      >
    >
"""


def write_project(
    project_path: Path,
    clips=(*ORIGINAL_CLIPS, NEW_VOICE_CLIP),
    render_bounds: int = 1,
    play_rate: float = 1,
) -> Path:
    project_path.write_text(
        f"""<REAPER_PROJECT 0.1 "7.39/win64" 1
  RENDER_RANGE {render_bounds} 0 0 0 1000
  PLAYRATE {play_rate} 0 0.25 4
  <TRACK
    NAME originals
{"".join(item_text(*clip) for clip in clips)}  >
>
""",
        encoding="utf-8",
    )
    return project_path


def test_every_original_clip_becomes_a_time_slot_of_the_render(tmp_path):
    project_path = write_project(tmp_path / "announcer.rpp")
    write_replacement_config(project_path, tmp_path / "renders" / "announcer.wav", tmp_path / "configs" / "config.json")
    rendered_file = "../renders/announcer.wav"
    assert json.loads((tmp_path / "configs" / "config.json").read_text()) == {
        "music/st1_lp.wav": {"replacement_file": rendered_file, "time_start": 0, "time_stop": 1.25},
        "sfx/allse/SND_SED_NAR_READY.wav": {"replacement_file": rendered_file, "time_start": 0, "time_stop": 1.25},
        "sfx/comn/SFXready_01_basic_macro.wav": {
            "replacement_file": rendered_file,
            "time_start": 1.5,
            "time_stop": 3.5,
        },
    }


@pytest.mark.parametrize(
    "project_settings",
    [
        {"play_rate": 1.048},
        {"render_bounds": 0},
        {"clips": (NEW_VOICE_CLIP,)},
        {"clips": (ORIGINAL_CLIPS[0], ORIGINAL_CLIPS[0])},
    ],
)
def test_project_whose_render_would_not_line_up_is_rejected(tmp_path, project_settings):
    project_path = write_project(tmp_path / "announcer.rpp", **project_settings)
    with pytest.raises(InvalidReplacementsError):
        write_replacement_config(project_path, tmp_path / "mix.wav", tmp_path / "config.json")
    assert not (tmp_path / "config.json").exists()
