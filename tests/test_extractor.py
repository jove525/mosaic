from mosaic.kb.extractor import extract_frames, FrameSet


def test_extract_frames_returns_frameset(tiny_video, tmp_dir):
    frame_dir = tmp_dir / "frames"
    result = extract_frames(tiny_video, frame_dir, interval_seconds=2)

    assert isinstance(result, FrameSet)
    assert len(result.frame_paths) >= 2  # 5s video at 2s interval = 2-3 frames
    assert all(p.exists() for p in result.frame_paths)
    assert all(p.suffix == ".png" for p in result.frame_paths)
    assert result.video_path == tiny_video


def test_extract_frames_skips_if_already_extracted(tiny_video, tmp_dir):
    frame_dir = tmp_dir / "frames"
    result1 = extract_frames(tiny_video, frame_dir, interval_seconds=2)
    result2 = extract_frames(tiny_video, frame_dir, interval_seconds=2)

    assert len(result1.frame_paths) == len(result2.frame_paths)
