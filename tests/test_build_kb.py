import subprocess
import sys


def test_build_kb_help():
    result = subprocess.run(
        [sys.executable, "build_kb.py", "--help"],
        capture_output=True, text=True, cwd="D:/Mosaic"
    )
    assert result.returncode == 0
    assert "--channel" in result.stdout
    assert "--limit" in result.stdout
