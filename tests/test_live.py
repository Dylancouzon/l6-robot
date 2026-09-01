import threading
import unittest
from unittest.mock import MagicMock, call, patch

import cv2

from robot.device.live import LiveApp


class LiveAppTests(unittest.TestCase):
    @patch("robot.device.live.sys.platform", "win32")
    @patch("robot.device.live.cv2.VideoCapture")
    def test_windows_camera_falls_back_from_directshow(self, video_capture):
        directshow = MagicMock()
        directshow.isOpened.return_value = False
        default = MagicMock()
        default.isOpened.return_value = True
        video_capture.side_effect = [directshow, default]

        LiveApp(MagicMock(), camera=2)

        self.assertEqual(video_capture.call_args_list, [
            call(2, cv2.CAP_DSHOW),
            call(2, cv2.CAP_ANY),
        ])
        directshow.release.assert_called_once_with()

    def test_voice_path_can_only_be_claimed_once(self):
        app = LiveApp.__new__(LiveApp)
        app.lock = threading.Lock()
        app.busy = False

        self.assertTrue(app._start_voice())
        self.assertFalse(app._start_voice())
        app._finish_voice()
        self.assertTrue(app._start_voice())


if __name__ == "__main__":
    unittest.main()
