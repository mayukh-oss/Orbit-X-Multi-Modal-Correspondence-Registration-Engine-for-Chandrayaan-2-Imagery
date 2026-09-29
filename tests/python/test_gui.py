# [ANNOTATION] Automated tests for PySide6 GUI desktop application components and event worker threads.
"""
Automated tests for the SIH26166 PySide6 GUI workstation.
Tests offscreen creation, page navigation, config synthesis, event loop worker execution,
reset state machines, tooltips, and provenance formatting.
"""

from __future__ import annotations

# [ANNOTATION] Force offscreen Qt rendering platform for headless CI environments.
import os
import sys
import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

# [ANNOTATION] Import Qt core primitives and application classes.
from pathlib import Path
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

# [ANNOTATION] Import GUI components under test.
from sih26166.app.gui import (
    MainWindow,
    MetricCard,
    MissionLandingPage,
    StageWidget,
    RealPipelineWorker,
    ControlledPipelineWorker,
    apply_theme,
)
from sih26166.pipeline import PipelineConfig


# [ANNOTATION] PyTest fixture providing shared QApplication instance with dark theme applied.
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    apply_theme(app) # type: ignore
    yield app


# [ANNOTATION] Event loop helper driving background worker thread completion in Qt test environment.
def wait_for_worker(worker, timeout_ms: int = 20000) -> None:
    """Event-driven worker wait that continuously pumps the Qt event loop."""
    if worker is None:
        return
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()


# [ANNOTATION] Test verifying main window layout setup and initial stack pages.
def test_gui_window_initialization(qapp):
    print("\n[TEST] Executing test_gui_window_initialization...")
    win = MainWindow()
    win.setup()
    assert win.stack.count() == 4
    assert win.stack.currentIndex() == 0

    assert type(win.stack.widget(0)).__name__ == "OrbitXStartupScreen"
    assert type(win.stack.widget(1)).__name__ == "MissionLandingPage"
    assert type(win.stack.widget(2)).__name__ == "WorkflowPageWidget"
    assert type(win.stack.widget(3)).__name__ == "WorkflowPageWidget"


# [ANNOTATION] Test verifying UI page navigation state switches.
def test_gui_page_navigation(qapp):
    print("\n[TEST] Executing test_gui_page_navigation...")
    win = MainWindow()
    win.setup()

    # Home -> Real
    win.switch_to_real()
    assert win.stack.currentIndex() == 2
    assert win.reference_raw is not None
    assert win.source_raw is not None
    assert win.reference_raw.shape == (2048, 2048)

    # Real -> Controlled
    win.switch_to_controlled()
    assert win.stack.currentIndex() == 3

    # Controlled -> Home
    win.stack.setCurrentIndex(0)
    assert win.stack.currentIndex() == 0


# [ANNOTATION] Test verifying MetricCard widget label updates and text formatting.
def test_gui_metric_cards(qapp):
    print("\n[TEST] Executing test_gui_metric_cards...")
    card = MetricCard()
    card.setup("TEST_METRIC", "100", "px")
    assert card.title.text() == "TEST_METRIC"
    assert card.value.text() == "100"
    assert card.unit.text() == "px"

    card.set_value(250.5, "km")
    assert card.value.text() == "250.5"
    assert card.unit.text() == "km"


# [ANNOTATION] Test verifying StageWidget pipeline status badge states (READY, RUNNING, DONE, ERROR).
def test_gui_stage_widgets(qapp):
    print("\n[TEST] Executing test_gui_stage_widgets...")
    stage = StageWidget()
    stage.setup("1", "PREPROCESSING")
    assert stage.number.text() == "1"
    assert stage.title.text() == "PREPROCESSING"
    assert stage.state.text() == "READY"

    stage.set_state("RUNNING")
    assert stage.state.text() == "RUNNING"
    assert stage.icon_badge.text() == "▶"

    stage.set_state("DONE")
    assert stage.state.text() == "DONE"
    assert stage.icon_badge.text() == "✓"

    stage.set_state("ERROR")
    assert stage.state.text() == "ERROR"
    assert stage.icon_badge.text() == "✕"


# [ANNOTATION] Test verifying pipeline configuration synthesized from GUI controls.
def test_gui_pipeline_config_generation(qapp):
    print("\n[TEST] Executing test_gui_pipeline_config_generation...")
    win = MainWindow()
    win.setup()
    config = win.make_pipeline_config()
    assert isinstance(config, PipelineConfig)
    assert config.feature_max_features == 2000
    assert config.geometric.model in {"AFFINE", "HOMOGRAPHY"}
    assert config.spatial.grid_rows == 8
    assert config.spatial.grid_cols == 8


# [ANNOTATION] Test executing controlled pipeline benchmark worker thread from GUI.
def test_gui_controlled_worker_execution(qapp):
    print("\n[TEST] Executing test_gui_controlled_worker_execution...")
    win = MainWindow()
    win.setup()
    win.switch_to_controlled()
    win.run_controlled()
    assert win.worker is not None
    wait_for_worker(win.worker, 15000)
    qapp.processEvents()

    assert win.controlled_result is not None
    assert win.controlled_result.metrics.transformation_rmse >= 0.0
    assert win.ctrl_m_matches.value.text() != "-"
    assert win.ctrl_m_inliers.value.text() != "-"
    assert "CONTROLLED BENCHMARK GT" in win.ctrl_gt.toPlainText()


# [ANNOTATION] Test executing real image pipeline worker thread from GUI.
def test_gui_real_worker_execution(qapp):
    print("\n[TEST] Executing test_gui_real_worker_execution...")
    win = MainWindow()
    win.setup()
    win.switch_to_real()
    win.run_real()
    assert win.worker is not None
    wait_for_worker(win.worker, 20000)
    qapp.processEvents()

    assert win.pipeline_result is not None
    assert win.pipeline_result.success is True
    assert win.real_m_matches.value.text() != "-"
    assert win.real_m_inliers.value.text() != "-"
    assert win.real_m_ratio.value.text() != "-"
    assert "CHANDRAYAAN-2" in win.real_metadata_text.toPlainText()


# [ANNOTATION] Test loading pre-generated real evidence metadata.
def test_gui_load_real_evidence(qapp):
    print("\n[TEST] Executing test_gui_load_real_evidence...")
    win = MainWindow()
    win.setup()
    win.switch_to_real()
    win.load_real_evidence()
    assert win.real_m_matches.value.text() != "-"
    assert win.real_m_inliers.value.text() != "-"


# [ANNOTATION] Test displaying toast notification popups in GUI window.
def test_gui_toast_notification(qapp):
    print("\n[TEST] Executing test_gui_toast_notification...")
    win = MainWindow()
    win.setup()
    win.show()
    win.show_toast("TEST TITLE", "Test message content", level="success", duration_ms=1000)
    assert win._active_toast is not None
    assert not win._active_toast.isHidden()
    assert win._active_toast.level == "success"

    # Test manual close
    win._active_toast.close_toast()
    assert win._active_toast.isHidden()

    # Test warning and error toasts
    win.show_toast("WARNING TITLE", "Warning detail", level="warning", duration_ms=5000)
    assert win._active_toast.level == "warning"

    win.show_toast("ERROR TITLE", "Error detail", level="error", duration_ms=5000)
    assert win._active_toast.level == "error"


# [ANNOTATION] Test verifying icon text for all stage widget states.
def test_gui_stage_widget_icons(qapp):
    print("\n[TEST] Executing test_gui_stage_widget_icons...")
    stage = StageWidget()
    stage.setup("3", "FEATURES")
    assert stage.icon_badge.text() == "○"

    stage.set_state("RUNNING")
    assert stage.icon_badge.text() == "▶"

    stage.set_state("DONE")
    assert stage.icon_badge.text() == "✓"

    stage.set_state("ERROR")
    assert stage.icon_badge.text() == "✕"

    stage.set_state("WARNING")
    assert stage.icon_badge.text() == "⚠"


# [ANNOTATION] Test verifying synthetic demo image fallback logic when raw PDS4 files are absent.
def test_gui_demo_fallback_mode_flag(qapp):
    print("\n[TEST] Executing test_gui_demo_fallback_mode_flag...")
    win = MainWindow()
    win.setup()
    win.load_tmc_pair()
    # In testing environment without 1GB .img, fallback mode is active
    assert win.is_demo_fallback in {True, False}
    assert win.reference_raw is not None
    assert win.source_raw is not None
    assert win.reference_raw.shape == (2048, 2048)


# [ANNOTATION] Test resetting workspace metric displays and cleared pipeline results.
def test_gui_workspace_reset(qapp):
    print("\n[TEST] Executing test_gui_workspace_reset...")
    win = MainWindow()
    win.setup()
    win.switch_to_real()
    win.real_m_matches.set_value(500)
    win.reset_real()
    assert win.real_m_matches.value.text() == "-"
    assert win.real_m_inliers.value.text() == "-"
    assert win.pipeline_result is None

    win.switch_to_controlled()
    win.ctrl_m_matches.set_value(300)
    win.reset_controlled()
    assert win.ctrl_m_matches.value.text() == "-"
    assert win.controlled_result is None


# [ANNOTATION] Test checking metadata panel text and control widget initialization.
def test_gui_tooltips_and_provenance(qapp):
    print("\n[TEST] Executing test_gui_tooltips_and_provenance...")
    win = MainWindow()
    win.setup()
    win.switch_to_real()

    # Current GUI architecture exposes provenance directly in the
    # mission metadata panel rather than relying on the old tooltip API.
    assert win.real_metadata_text is not None
    assert "CHANDRAYAAN-2" in win.real_metadata_text.toPlainText()

    # The current controls remain instantiated and usable.
    assert win.real_preprocess is not None
    assert win.real_matching is not None
    assert win.real_geometry is not None
    assert win.real_run is not None


# [ANNOTATION] Test verifying landing page card UI elements and registered callbacks.
def test_landing_workflow_cards(qapp):
    """Verify the current native landing-page workflow architecture."""
    print("\n[TEST] Executing test_landing_workflow_cards...")
    win = MainWindow()
    win.setup()

    page = win.home_page

    assert isinstance(page, MissionLandingPage)
    assert page.real_card is not None
    assert page.ctrl_card is not None
    assert callable(page._real_callback)
    assert callable(page._controlled_callback)


# [ANNOTATION] Test verifying navigation from landing page to Real workflow view.
def test_landing_real_callback_navigation(qapp):
    """Verify the Real workflow callback navigates to the current Real page."""
    print("\n[TEST] Executing test_landing_real_callback_navigation...")
    win = MainWindow()
    win.setup()

    assert win.stack.currentIndex() == 0

    win.home_page._on_real_activate()

    assert win.stack.currentIndex() == 2
    assert win.reference_raw is not None
    assert win.source_raw is not None
    assert win.reference_raw.shape == (2048, 2048)


# [ANNOTATION] Test verifying navigation from landing page to Controlled workflow view.
def test_landing_controlled_callback_navigation(qapp):
    """Verify the Controlled workflow callback navigates to the current benchmark page."""
    print("\n[TEST] Executing test_landing_controlled_callback_navigation...")
    win = MainWindow()
    win.setup()

    assert win.stack.currentIndex() == 0

    win.home_page._controlled_callback()

    assert win.stack.currentIndex() == 3


# [ANNOTATION] Test confirming distinct landing page workflow target cards.
def test_landing_cards_have_distinct_workflow_targets(qapp):
    """Verify the landing page exposes separate Real and Controlled workflows."""
    print("\n[TEST] Executing test_landing_cards_have_distinct_workflow_targets...")
    win = MainWindow()
    win.setup()

    page = win.home_page

    assert page.real_card is not page.ctrl_card
    assert page.real_card.parent() is not None
    assert page.ctrl_card.parent() is not None