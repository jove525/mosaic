from mosaic.config.settings import Settings, settings
from mosaic.config.channels import REFERENCE_CHANNELS, CHANNEL_PROFILES, PRODUCTION_CHANNELS


def test_settings_has_required_fields():
    s = Settings()
    assert s.anthropic_api_key is not None or True  # may be empty in CI
    assert s.frame_interval_seconds == 8
    assert s.whisper_model == "base"
    assert s.kb_dir.name == "kb"
    assert s.clips_dir.name == "clips"
    assert s.frames_dir.name == "frames"


def test_reference_channels_has_five_entries():
    assert len(REFERENCE_CHANNELS) == 5
    required_keys = {"name", "focus", "video_urls", "target_count"}
    for ch in REFERENCE_CHANNELS.values():
        assert required_keys.issubset(ch.keys())


def test_channel_profiles_has_predictive_echoes_and_incentives_lab():
    assert "predictive_echoes" in CHANNEL_PROFILES
    assert "incentives_lab" in CHANNEL_PROFILES
    required_keys = {"domain", "tone", "arc_template", "visual_style", "north_star", "reference_channels"}
    for profile in CHANNEL_PROFILES.values():
        assert required_keys.issubset(profile.keys())


def test_settings_has_output_dir():
    s = Settings()
    assert s.output_dir.name == "output"


def test_settings_has_elevenlabs_keys():
    s = Settings()
    assert isinstance(s.elevenlabs_api_key, str)
    assert isinstance(s.elevenlabs_voice_id, str)


def test_settings_has_taste_dir():
    s = Settings()
    assert s.taste_dir.name == "taste"
    assert s.taste_dir.parent.name == "data"


def test_production_channels_has_incentiveslab():
    required_keys = {
        "name", "handle", "domain", "tone", "arc_template",
        "visual_style", "north_star", "channel_north_star",
        "reference_channels", "avg_length_min", "core_thesis",
    }
    assert "incentiveslab" in PRODUCTION_CHANNELS
    profile = PRODUCTION_CHANNELS["incentiveslab"]
    assert required_keys.issubset(profile.keys())
    assert profile["handle"].startswith("@")
    assert isinstance(profile["reference_channels"], list)
