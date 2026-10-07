from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def test_replayed_frame_carries_the_array_sensitivity_and_geometry(rclpy_context, tmp_path: Path) -> None:
    from arena_robots.audio import load_array_spec

    from arena_hearing.audio_replay import AudioReplay

    array = load_array_spec("four_mic")
    wav = tmp_path / "four.wav"
    wavfile.write(wav, 16000, np.zeros((640, 4), dtype=np.int16))
    node = AudioReplay(str(wav), "replay_test/audio/raw_array", 320, 0.0, "jackal/base_link", False, array)
    try:
        msg = node.frame(node.audio[:320])
    finally:
        node.destroy_node()

    assert msg.sensitivity_dbfs_at_94_dbspl == array.sensitivity_dbfs_at_94_dbspl
    assert msg.sensitivity_dbfs_at_94_dbspl != 0.0
    assert list(msg.channel_names) == list(array.channel_names)
    assert (msg.channel_count, msg.frame_count) == (4, 320)
