# MLX Speech Daemon

`msd` is a manual-session, TTS-only daemon for MLX-Audio Qwen3 CustomVoice models.

V1 exposes one public command:

```bash
msd say "Hello." --voice Aiden --instruct "clear, warm, brisk"
msd up --model cv17-q8
msd status
msd render --text "Offline file." --output out.wav
msd hermes --input /tmp/in.txt --output /tmp/out.wav
msd stop
msd down
msd serve --foreground
```

The daemon is started automatically by `say`, `up`, and `hermes` when needed. It listens on a Unix socket, keeps one selected CustomVoice model warm, streams generated chunks to the default audio output for `say`, and interrupts active playback by default when a newer `say` request arrives.

V1 intentionally does not include STT, launchd, cron, containers, VMs, an autostart gateway, or Hermes source patches.

## Install

```bash
uv tool install --force --editable ".[dev]"
```

If `~/.local/bin` is on `PATH`, this exposes `msd`.

MP3 output requires `ffmpeg` on `PATH`. WAV output does not.

## Model Aliases

```text
cv17-q8 -> mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
cv17-q6 -> mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit
cv06-q8 -> mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
```

Aliases and speaker names are case-insensitive.

## Test Doubles

The real engine is used by default. Tests and dry smokes can use:

```bash
MSD_ENGINE=fake MSD_PLAYBACK=fake msd serve --foreground
```
