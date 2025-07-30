import pytest
from PyQt6.QtWidgets import QApplication
from diffmap_view.gui.windows.diffmap_view_window import diffmap_viewWindow

@pytest.fixture(scope='module')
def app():
    import sys
    app = QApplication(sys.argv)
    yield app
    app.quit()

def test_window_starts(app):
    window = diffmap_viewWindow()
    assert window is not None
    assert window.isVisible() is False  # should not be visible until shown
