import unittest

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_app_starts_without_exception(self):
        app = AppTest.from_file("streamlit_app.py")
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.title) + len(app.markdown), 1)


if __name__ == "__main__":
    unittest.main()
