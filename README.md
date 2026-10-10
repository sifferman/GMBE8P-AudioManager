# smbaudio

Replace the music and sound effects in Super Monkey Ball (GameCube, NTSC-U, game ID `GMBE8P`).

This tool can dump all of the game's audio to WAV. Then, you can write a JSON file that tells the tool to inject your own audio files, and build a modified ISO.

## Install

```bash
pipx install git+https://github.com/sifferman/GMBE8P-AudioManager
```

Install [ffmpeg](https://ffmpeg.org/) as well if you want to use MP3, OGG, or other formats as replacements. Without ffmpeg, replacements must be WAV or FLAC.

## Use

Dump every music track and sound effect:

```bash
smbaudio dump "Super Monkey Ball (USA).iso" -o dumped_audio
```

Example JSON file that lists a mapping of songs to be replaced with the file at the specified path. Paths are relative to the JSON file's folder:

```json
{
  "music/adv_lp.wav": "songs/replacement_dong.mp3",
  "sfx/comn/SFX1up_basic_macro.wav": "sounds/replacement_sfx.wav"
}
```

A replacement can also be one part of a longer file. Times are in seconds; `time_start` defaults to 0 and `time_stop` defaults to the end of the file:

```json
{
  "sfx/comn/SFXready_01_basic_macro.wav": { "replacement_file": "mix.wav", "time_start": 12.5, "time_stop": 13.8 },
  "sfx/comn/SFXgo_01_basic_macro.wav": { "replacement_file": "mix.wav", "time_start": 13.8 }
}
```

Then build the modified ISO:

```bash
smbaudio patch --json replacements.json -i "Super Monkey Ball (USA).iso" -o modified.iso
```

## DAW Integration

This project has Reaper integration. `smbaudio reaper2json` will convert a .rpp file into a .json file, by mapping evey track that matches the filename of an original dumped filename to a start and stop time. If you mute the original dumped file, then you can record over it, render the project, and the .json will grab your audio instead.

```bash
# first, render the entire project.rpp into mix.wav. then run
smbaudio reaper2json -i project.rpp --render mix.wav -o replacements.json
smbaudio patch --json replacements.json -i "Super Monkey Ball (USA).iso" -o modified.iso
```
