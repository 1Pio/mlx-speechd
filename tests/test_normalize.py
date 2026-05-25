from mlx_speechd.normalize import normalize_model, normalize_voice, speed_to_instruction


def test_model_aliases_are_case_insensitive() -> None:
    model_id, alias = normalize_model("CV17-Q8")

    assert alias == "cv17-q8"
    assert model_id == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"


def test_model_ids_pass_through() -> None:
    model_id, alias = normalize_model("mlx-community/custom")

    assert model_id == "mlx-community/custom"
    assert alias is None


def test_voice_names_are_case_insensitive() -> None:
    assert normalize_voice("aiden") == "Aiden"
    assert normalize_voice("UNCLE_FU") == "Uncle_Fu"


def test_speed_is_folded_into_instruction() -> None:
    assert speed_to_instruction("calm", 0.85) == "calm, speaking slower, approximate speed 0.85"
    assert speed_to_instruction("calm", 1.0) == "calm"
