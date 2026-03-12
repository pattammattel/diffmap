import pytest
from PyQt6.QtWidgets import QApplication
from diffmap.gui.windows.diffmap_view_window import DiffMapWindow

@pytest.fixture(scope='module')
def app():
    import sys
    app = QApplication(sys.argv)
    yield app
    app.quit()

def test_window_starts(app):
    window = DiffMapWindow()
    assert window is not None
    assert window.isVisible() is False  # should not be visible until shown
