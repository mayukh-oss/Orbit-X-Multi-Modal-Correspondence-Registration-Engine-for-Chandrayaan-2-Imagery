# [ANNOTATION] Unit tests for GUI image conversion and difference visualization helpers.
"""
Tests for GUI visual rendering functions used by the final SIH26166 GUI.
"""

from __future__ import annotations

# [ANNOTATION] Force offscreen Qt rendering platform.
import os
import sys

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap

# [ANNOTATION] Import GUI visual rendering functions under test.
from sih26166.app.gui import (
    array_to_pixmap,
    make_difference_visual,
    apply_theme,
)


# [ANNOTATION] Shared QApplication fixture.
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    apply_theme(app) # type: ignore
    yield app


# [ANNOTATION] Test converting a 2D NumPy array into a scaled QPixmap for UI display.
def test_array_to_pixmap(qapp):
    print("\n[TEST] Executing test_array_to_pixmap...")
    img = np.zeros((100, 100), dtype=np.uint8)

    pix = array_to_pixmap(img, 200, 200)

    assert isinstance(pix, QPixmap)
    assert not pix.isNull()
    assert pix.width() == 200
    assert pix.height() == 200


# [ANNOTATION] Test converting random high-contrast image array into QPixmap.
def test_array_to_pixmap_textured_image(qapp):
    print("\n[TEST] Executing test_array_to_pixmap_textured_image...")
    np.random.seed(26166)

    img = np.random.randint(
        0,
        256,
        (256, 256),
        dtype=np.uint8,
    )

    pix = array_to_pixmap(img, 512, 512)

    assert isinstance(pix, QPixmap)
    assert not pix.isNull()
    assert pix.width() == 512
    assert pix.height() == 512


# [ANNOTATION] Test creating false-color difference map between reference and registered images.
def test_make_difference_visual(qapp):
    print("\n[TEST] Executing test_make_difference_visual...")
    reference = np.zeros((256, 256), dtype=np.uint8)
    registered = np.zeros((256, 256), dtype=np.uint8)

    registered[100:150, 100:150] = 255

    pix = make_difference_visual(
        reference,
        registered,
        512,
        512,
    )

    assert isinstance(pix, QPixmap)
    assert not pix.isNull()
    assert pix.width() == 512
    assert pix.height() == 512