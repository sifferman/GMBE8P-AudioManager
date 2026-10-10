# smbaudio

Replace the music and sound effects in Super Monkey Ball (GameCube, NTSC-U, game ID `GMBE8P`).

This tool can dump all of the game's audio to WAV. Then, you can write a JSON file that tells the tool to inject your own audio files, and build a modified ISO.

## Install

```
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

Then build the modified ISO:

```bash
smbaudio patch --json replacements.json -i "Super Monkey Ball (USA).iso" -o modified.iso
```
