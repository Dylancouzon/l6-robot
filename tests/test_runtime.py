import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from robot.device.runtime import UTTERANCE_WAV, stamp, when


class RuntimePortabilityTests(unittest.TestCase):
    def setUp(self):
        # Fixed local-time structs make these tests independent of the machine's
        # timezone and exercise formatting without Unix-only %-d/%-I flags.
        self.morning = time.struct_time((2026, 8, 3, 9, 7, 0, 0, 215, -1))
        self.evening = time.struct_time((2026, 8, 2, 21, 5, 0, 6, 214, -1))

    def test_spoken_time_today_has_no_leading_zero(self):
        with patch("robot.device.runtime.time.localtime",
                   side_effect=[self.morning, self.morning]):
            self.assertEqual(when(1, now=2), "9:07 AM")

    def test_spoken_time_on_an_older_day(self):
        with patch("robot.device.runtime.time.localtime",
                   side_effect=[self.evening, self.morning]):
            self.assertEqual(when(1, now=2), "Aug 2, 9:05 PM")

    def test_ui_stamp(self):
        with patch("robot.device.runtime.time.localtime",
                   return_value=self.morning):
            self.assertEqual(stamp(1), "Aug 3, 09:07")

    def test_recording_uses_the_platform_temp_directory(self):
        self.assertEqual(Path(UTTERANCE_WAV).parent,
                         Path(tempfile.gettempdir()))
        self.assertIn(str(os.getpid()), Path(UTTERANCE_WAV).stem)


if __name__ == "__main__":
    unittest.main()
