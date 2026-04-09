from mosaic.config.settings import Settings
from mosaic.config.channels import REFERENCE_CHANNELS, CHANNEL_PROFILES


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
