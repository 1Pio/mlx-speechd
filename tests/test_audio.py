import wave

import numpy as np

from mlx_speechd.audio import write_audio_file


def test_write_wav(tmp_path) -> None:
    output = tmp_path / "out.wav"
    samples = np.zeros(2400, dtype=np.float32)

    write_audio_file(output, samples, 24000, "wav")

    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 2400
