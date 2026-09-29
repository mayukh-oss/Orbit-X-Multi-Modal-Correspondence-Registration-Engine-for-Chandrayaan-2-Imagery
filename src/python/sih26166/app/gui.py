# [ANNOTATION] Module docstring describing the Chandrayaan-2 Image Correspondence Workstation GUI.
"""
SIH26166 — Chandrayaan-2 Image Correspondence Workstation
SMART INDIA HACKATHON 2026 • ISRO Problem Statement SIH26166

Team Orbit X — Multi-modal, Sun angle and scale invariant image correspondence
using Chandrayaan-2 optical images (OHRC, TMC and IIRS).

Scientific Desktop Application Architecture:
1. Real Chandrayaan-2 TMC-2 (NCA/NCF) Image Correspondence & Registration
2. Controlled Validation Benchmark with Known Ground-Truth Evaluation
"""

# [ANNOTATION] Enable future annotations behavior for modern type hinting support.
from __future__ import annotations

# [ANNOTATION] Import standard library modules for system interaction, path manipulation, typing, and JSON.
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# [ANNOTATION] Import Shiboken6 bindings to check C++ underlying object validity.
import shiboken6

# [ANNOTATION] Import OpenCV for computer vision tasks and image transformations.
import cv2

# [ANNOTATION] Import NumPy for high-performance multidimensional array manipulations.
import numpy as np

# [ANNOTATION] Import core Qt primitives and types from PySide6 for GUI design.
from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QThread,
    QTimer,
    QUrl,
    QVariantAnimation,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# [ANNOTATION] Attempt to import PySide6 QtMultimedia audio components for UI sound effects.
try:
    from PySide6.QtMultimedia import QSoundEffect
    HAS_MULTIMEDIA = True
    print("[AUDIO] QtMultimedia module detected and loaded successfully.")
except Exception as exc:
    HAS_MULTIMEDIA = False
    print(f"[AUDIO] QtMultimedia unavailable ({exc}). Audio effects will be disabled.")

# [ANNOTATION] Import core scientific pipeline components for feature extraction, matching, and geometry.
from sih26166.correspondence.features import detect_features
from sih26166.correspondence.matching import MatchingConfig, match_features
from sih26166.correspondence.types import CorrespondenceSet
from sih26166.evaluation.metrics import (
    spatial_coverage,
    transformation_residual_metrics,
)
from sih26166.experiments.benchmark import (
    BenchmarkResult,
    run_controlled_benchmark,
)
from sih26166.io.image_reader import read_image
from sih26166.io.pds4 import PDS4ImageMetadata, read_pds4_label
from sih26166.io.tmc2 import TMC2Product, get_tmc2_20200804_pair
from sih26166.paths import (
    INTERIM_DIR,
    PROJECT_ROOT,
    SAMPLES_DIR,
)
from sih26166.pipeline import (
    PipelineConfig,
    PipelineResult,
    _apply_transformation,
)
from sih26166.preprocessing import (
    PreprocessingConfig,
    percentile_normalize,
    preprocess_for_features,
)
from sih26166.refinement.subpixel import (
    SubpixelRefinementConfig,
    refine_correspondences,
)
from sih26166.registration.register import (
    RegistrationConfig,
    register_image,
)
from sih26166.spatial.distribution import (
    SpatialSelectionConfig,
    select_spatially_distributed,
)
from sih26166.verification.geometric import (
    GeometricVerificationConfig,
    verify_geometry,
)

# =====================================================================
# Centralized Semantic Palette & Design Tokens
# =====================================================================
# [ANNOTATION] Define color constant hex codes for UI styling and thematic consistency across widgets.
COLOR_BG_MAIN = "#050505"
COLOR_SURFACE_PANEL = "#0a0a0a"
COLOR_SURFACE_ELEVATED = "#111111"
COLOR_SURFACE_CARD = "#0a0a0a"
COLOR_BORDER_DEFAULT = "#333333"
COLOR_BORDER_SUBTLE = "#1a1a1a"
COLOR_BORDER_HOVER = "#F3B718"

COLOR_TEXT_PRIMARY = "#ffffff"
COLOR_TEXT_SECONDARY = "#a0a0a0"
COLOR_TEXT_MUTED = "#666666"
COLOR_TEXT_CODE = "#F3B718"

COLOR_SUCCESS = "#4caf50"
COLOR_SUCCESS_BG = "#0d1f10"
COLOR_SUCCESS_BORDER = "#2e7d32"
COLOR_SUCCESS_TEXT = "#81c784"

COLOR_ERROR = "#f44336"
COLOR_ERROR_BG = "#2a0d0d"
COLOR_ERROR_BORDER = "#d32f2f"
COLOR_ERROR_TEXT = "#e57373"

COLOR_WARNING = "#ff9800"
COLOR_WARNING_BG = "#261700"
COLOR_WARNING_BORDER = "#f57c00"
COLOR_WARNING_TEXT = "#ffb74d"

COLOR_CYAN = "#F3B718"
COLOR_CYAN_ACCENT = "#d4ac0d"
COLOR_CYAN_BG = "#1a1505"
COLOR_CYAN_BORDER = "#b88a12"
COLOR_CYAN_TEXT = "#fcd153"

COLOR_CONTROLLED = "#e0e0e0"
COLOR_CONTROLLED_BG = "#1a1a1a"
COLOR_CONTROLLED_BORDER = "#666666"
COLOR_CONTROLLED_TEXT = "#ffffff"

COLOR_DISABLED = "#475569"
COLOR_DISABLED_BG = "#0e0f14"
COLOR_DISABLED_BORDER = "#1a1b22"

# [ANNOTATION] Default bounding window tuple definitions for Chandrayaan-2 TMC-2 camera view (x, y, w, h).
NCA_WINDOW = (576, 73576, 2048, 2048)
NCF_WINDOW = (476, 93676, 2048, 2048)

# [ANNOTATION] Named string representations of the internal processing pipeline stages.
REAL_STAGE_NAMES = [
    "DATA INGESTION",
    "PREPROCESS",
    "FEATURES",
    "MATCHING",
    "GEOMETRY",
    "SPATIAL / SUBPX",
    "REGISTER",
]

# [ANNOTATION] Formatted stage labels displayed in the UI sidebar.
NEW_VISUAL_STAGE_NAMES = [
    "01 INPUT",
    "02 CORRESPONDENCE",
    "03 SPATIAL DIST",
    "04 REGISTRATION",
    "05 VALIDATION",
]

# =====================================================================
# Visual Rendering Utilities
# =====================================================================

def array_to_pixmap(image: np.ndarray, width: int = 0, height: int = 0) -> QPixmap:
    # [ANNOTATION] Utility function to convert a NumPy image matrix into a PySide6 QPixmap for display.
    print(f"[RENDER] Converting NumPy array shape {image.shape} to QPixmap...")
    arr = np.asarray(image)
    if arr.size == 0:
        print("[RENDER] Warning: Empty NumPy array passed. Returning null QPixmap.")
        return QPixmap()

    # [ANNOTATION] Perform percentile normalization if input datatype is not 8-bit unsigned integer.
    if arr.dtype != np.uint8:
        print(f"[RENDER] Array dtype is {arr.dtype}. Normalizing to uint8...")
        arr = percentile_normalize(arr)

    # [ANNOTATION] Format channels appropriately to produce a 3-channel RGB matrix.
    if arr.ndim == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        rgb = cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr.copy()
    else:
        rgb = cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2RGB)

    # [ANNOTATION] Extract image dimensions and construct QImage buffer.
    h, w = rgb.shape[:2]
    qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(qimage)

    # [ANNOTATION] Scale pixmap if target dimensions are requested.
    if width > 0 and height > 0:
        print(f"[RENDER] Scaling pixmap to {width}x{height} keeping aspect ratio.")
        return pm.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return pm

def make_difference_visual(reference: np.ndarray, registered: np.ndarray, width: int = 0, height: int = 0) -> QPixmap:
    # [ANNOTATION] Compute absolute difference map between reference and registered imagery to generate false-color heatmaps.
    print("[RENDER] Generating visual error/difference map heatmap...")
    a = percentile_normalize(reference).astype(np.int16)
    b = percentile_normalize(registered).astype(np.int16)

    # [ANNOTATION] Determine overlapping dimensions and calculate absolute pixelwise intensity error.
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    diff = np.abs(a[:h, :w] - b[:h, :w]).astype(np.uint8)

    # [ANNOTATION] Apply inferno colormap to visualize alignment errors intuitively.
    colored_diff = cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    h2, w2 = colored_diff.shape[:2]
    qimage = QImage(colored_diff.data, w2, h2, colored_diff.strides[0], QImage.Format.Format_RGB888).copy()
    pm = QPixmap.fromImage(qimage)

    # [ANNOTATION] Scale if requested.
    if width > 0 and height > 0:
        return pm.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return pm

# =====================================================================
# Scientific Viewport Components
# =====================================================================

class ViewModeSelector(QFrame):
    """Small technical mode selector for viewport views."""
    # [ANNOTATION] Declare PySide signal emitted when a viewport mode button is clicked.
    mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        print("[UI_COMPONENT] Constructing ViewModeSelector widget...")
        self.setObjectName("modeSelector")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.buttons = {}
        # [ANNOTATION] List of interactive view modes provided in the UI header.
        modes = [
            ("REFERENCE", "A) REFERENCE"),
            ("SOURCE", "B) SOURCE"),
            ("CORRESPONDENCE", "C) CORRESPONDENCE"),
            ("SPATIAL", "D) SPATIAL GRID"),
            ("REGISTERED", "E) REGISTERED"),
            ("DIFFERENCE", "F) DIFFERENCE"),
        ]

        # [ANNOTATION] Create toggle pushbuttons for each display mode.
        for m, label in modes:
            btn = QPushButton(label)
            btn.setObjectName("modeButton")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m_id=m: self._on_btn_clicked(m_id))
            self.buttons[m] = btn
            layout.addWidget(btn)

        layout.addStretch()
        self.set_mode("REFERENCE")

    def _on_btn_clicked(self, mode: str):
        # [ANNOTATION] Handler triggered when user selects a viewport mode button.
        print(f"[UI_EVENT] ViewModeSelector button clicked: {mode}")
        self.set_mode(mode)
        self.mode_changed.emit(mode)

    def set_mode(self, mode: str):
        # [ANNOTATION] Update button checked state based on current active view mode.
        for m, btn in self.buttons.items():
            btn.setChecked(m == mode)

class InteractiveImageWorkspace(QFrame):
    """Cinematic interactive central image workspace with pan, zoom, wipes, and telemetry."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        print("[UI_COMPONENT] Initializing InteractiveImageWorkspace...")
        self.setObjectName("imageWorkspace")
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # [ANNOTATION] State properties holding pixmaps, execution results, and transform vectors.
        self.mode = "REFERENCE"
        self.ref_pixmap = QPixmap()
        self.src_pixmap = QPixmap()
        self.reg_pixmap = QPixmap()
        self.diff_pixmap = QPixmap()
        self.pipeline_result: PipelineResult | None = None
        self.benchmark_result: BenchmarkResult | None = None

        # [ANNOTATION] Viewport manipulation limits and state tracking.
        self.pan = QPointF(0, 0)
        self.zoom = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 30.0

        self._panning = False
        self._wiping = False
        self.wipe_ratio = 0.5
        self._last_mouse = QPointF()
        self._hover_pt = QPointF(-1, -1)

    def set_mode(self, mode: str):
        # [ANNOTATION] Change view mode and automatically readjust scale if transitioning correspondence overlays.
        print(f"[WORKSPACE] Setting workspace view mode to: {mode}")
        old = self.mode
        self.mode = mode
        if (old == "CORRESPONDENCE" and mode != "CORRESPONDENCE") or (old != "CORRESPONDENCE" and mode == "CORRESPONDENCE"):
            self.fit_to_view()
        self.update()

    def set_data(self, ref: QPixmap, src: QPixmap, reg: QPixmap, diff: QPixmap, result=None, benchmark=None):
        # [ANNOTATION] Store updated rendered graphics and operational pipeline results.
        print("[WORKSPACE] Storing new pixmap buffers and analytical pipeline results...")
        self.ref_pixmap = ref
        self.src_pixmap = src
        self.reg_pixmap = reg
        self.diff_pixmap = diff
        self.pipeline_result = result
        self.benchmark_result = benchmark
        self.fit_to_view()

    def fit_to_view(self):
        # [ANNOTATION] Automatically calculate zoom factor and offset to center imagery in viewport.
        if self.ref_pixmap.isNull():
            return
        
        print("[WORKSPACE] Recalculating scale and positioning to fit image viewport...")
        lw = self.ref_pixmap.width() * 2 if self.mode == "CORRESPONDENCE" else self.ref_pixmap.width()
        lh = self.ref_pixmap.height()

        zx = self.width() / lw if lw > 0 else 1
        zy = self.height() / lh if lh > 0 else 1

        self.zoom = min(zx, zy) * 0.95
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom))

        self.pan = QPointF((self.width() - lw * self.zoom) / 2, (self.height() - lh * self.zoom) / 2)
        self.update()

    def wheelEvent(self, event):
        # [ANNOTATION] Intercept mouse wheel events to zoom in or out centered at cursor position.
        angle = event.angleDelta().y()
        factor = 1.15 if angle > 0 else 0.85
        
        old_pos = event.position()
        ix = (old_pos.x() - self.pan.x()) / self.zoom
        iy = (old_pos.y() - self.pan.y()) / self.zoom

        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        
        self.pan = QPointF(old_pos.x() - ix * self.zoom, old_pos.y() - iy * self.zoom)
        self.update()

    def mousePressEvent(self, event):
        # [ANNOTATION] Handle mouse button clicks for panning viewport or adjusting swipe wipe bar.
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == "REGISTERED" and not self.ref_pixmap.isNull():
                hx = self.pan.x() + self.ref_pixmap.width() * self.wipe_ratio * self.zoom
                if abs(event.position().x() - hx) < 20:
                    print("[WORKSPACE] User engaged registration wipe bar.")
                    self._wiping = True
                    return
            self._panning = True
            self._last_mouse = event.position()

    def mouseMoveEvent(self, event):
        # [ANNOTATION] Track cursor hover coordinates and perform real-time pan/wipe updates.
        pos = event.position()
        self._hover_pt = QPointF((pos.x() - self.pan.x()) / self.zoom, (pos.y() - self.pan.y()) / self.zoom)

        if self._panning:
            self.pan += (pos - self._last_mouse)
            self._last_mouse = pos
        elif self._wiping and not self.ref_pixmap.isNull():
            nw = (pos.x() - self.pan.x()) / (self.ref_pixmap.width() * self.zoom)
            self.wipe_ratio = max(0.0, min(1.0, nw))
            
        self.update()

    def mouseReleaseEvent(self, event):
        # [ANNOTATION] Release panning/wiping interaction state flags.
        self._panning = False
        self._wiping = False

    def mouseDoubleClickEvent(self, event):
        # [ANNOTATION] Double click resets viewport orientation.
        print("[WORKSPACE] Double click registered: Fitting view to window bounds...")
        self.fit_to_view()

    def paintEvent(self, event):
        # [ANNOTATION] Render core image layers, correspondence vectors, grid indicators, and telemetry.
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.fillRect(self.rect(), QColor(5, 5, 5))

            if self.ref_pixmap.isNull():
                painter.setPen(QColor(80, 80, 80))
                painter.setFont(QFont("Consolas", 11))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "NO DATA / AWAITING MISSION PAYLOAD")
                return

            painter.save()
            painter.translate(self.pan)
            painter.scale(self.zoom, self.zoom)

            w, h = self.ref_pixmap.width(), self.ref_pixmap.height()

            from PySide6.QtCore import QLineF
            
            # [ANNOTATION] Render image according to active view mode.
            if self.mode in ("REFERENCE", "SPATIAL"):
                painter.drawPixmap(0, 0, self.ref_pixmap)
            elif self.mode == "SOURCE":
                painter.drawPixmap(0, 0, self.src_pixmap)
            elif self.mode == "DIFFERENCE":
                painter.drawPixmap(0, 0, self.diff_pixmap)
            elif self.mode == "CORRESPONDENCE":
                painter.drawPixmap(0, 0, self.ref_pixmap)
                if not self.src_pixmap.isNull():
                    painter.drawPixmap(w, 0, self.src_pixmap)

                res = self.pipeline_result
                if res:
                    m = res.initial_matches
                    mask = res.geometric_verification.inlier_mask
                    limit = min(len(m.reference_points), 350)
                    
                    inliers, outliers = [], []
                    for i in range(limit):
                        line = QLineF(m.reference_points[i][0], m.reference_points[i][1], 
                                      m.source_points[i][0] + w, m.source_points[i][1])
                        if i < len(mask) and mask[i]:
                            inliers.append(line)
                        else:
                            outliers.append(line)

                    # [ANNOTATION] Draw geometric outliers in red and verified inliers in green.
                    painter.setPen(QPen(QColor(239, 68, 68, 60), max(1.0, 1.0/self.zoom)))
                    painter.drawLines(outliers)
                    painter.setPen(QPen(QColor(16, 185, 129, 200), max(1.0, 1.5/self.zoom)))
                    painter.drawLines(inliers)
                    
                    rad = max(1.0, 2.0 / self.zoom)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(16, 185, 129, 255))
                    for l in inliers:
                        painter.drawEllipse(l.p1(), rad, rad)
                        painter.drawEllipse(l.p2(), rad, rad)

            elif self.mode == "REGISTERED":
                # [ANNOTATION] Render split-screen wipe view blending reference and registered images.
                wx = int(w * self.wipe_ratio)
                if wx > 0: painter.drawPixmap(0, 0, self.ref_pixmap, 0, 0, wx, h)
                if wx < w: painter.drawPixmap(wx, 0, self.reg_pixmap, wx, 0, w - wx, h)
                painter.setPen(QPen(QColor(243, 183, 24, 200), max(1.0, 2.0/self.zoom)))
                painter.drawLine(wx, 0, wx, h)

            if self.mode == "SPATIAL" and self.pipeline_result:
                # [ANNOTATION] Overlay spatial distribution grid and selected correspondence points.
                rws, cls = 8, 8
                rs, cs = h / rws, w / cls
                painter.setPen(QPen(QColor(90, 110, 130, 140), max(1.0, 1.0/self.zoom)))
                lines = [QLineF(0, r*rs, w, r*rs) for r in range(1, rws)] + [QLineF(c*cs, 0, c*cs, h) for c in range(1, cls)]
                painter.drawLines(lines)

                pts = self.pipeline_result.spatial_selection.correspondences.reference_points
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(230, 210, 40, 220))
                rad = max(1.0, 3.0 / self.zoom)
                for p in pts:
                    painter.drawEllipse(QPointF(p[0], p[1]), rad, rad)

                hx, hy = self._hover_pt.x(), self._hover_pt.y()
                if 0 <= hx < w and 0 <= hy < h:
                    painter.setBrush(QColor(243, 183, 24, 50))
                    painter.drawRect(QRectF(int(hx/cs)*cs, int(hy/rs)*rs, cs, rs))

            painter.restore()

            # [ANNOTATION] Render telemetry overlay showing mode, wipe ratio, zoom level, and pixel coordinates.
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            
            if self.mode == "REGISTERED":
                sx = self.pan.x() + w * self.wipe_ratio * self.zoom
                painter.setPen(QColor(243, 183, 24))
                painter.drawText(int(sx) + 8, self.height() - 20, f"WIPE {int(self.wipe_ratio*100)}%")

            painter.setPen(QColor(243, 183, 24))
            painter.drawText(10, 20, f"VIEWPORT MODE // {self.mode}")

            painter.setPen(QColor(148, 163, 184))
            text = f"ZOOM {self.zoom:.2f}x   X: {int(self._hover_pt.x()):04d}  Y: {int(self._hover_pt.y()):04d}" if self._hover_pt.x() >= 0 else f"ZOOM {self.zoom:.2f}x   X: ----  Y: ----"
            rect = painter.fontMetrics().boundingRect(text)
            painter.drawText(self.width() - rect.width() - 15, self.height() - 15, text)
            
        finally:
            painter.end()

# =====================================================================
# Toast Notification Overlay
# =====================================================================

class ToastNotification(QFrame):
    """Temporary non-blocking status notification overlay."""
    def __init__(self, parent: QWidget, title: str, message: str, level: str = "info", duration_ms: int = 4000) -> None:
        super().__init__(parent)
        print(f"[TOAST] Triggering {level.upper()} toast notification: '{title}' - '{message}'")
        self.level = level
        self.setObjectName(f"toast_{level}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        icons = {"success": "✓", "warning": "⚠", "error": "✕", "info": "ℹ"}
        icon_lbl = QLabel(icons.get(level, "ℹ"))
        icon_lbl.setObjectName(f"toastIcon_{level}")

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("toastTitle")

        msg_lbl = QLabel(message)
        msg_lbl.setObjectName("toastMessage")
        msg_lbl.setWordWrap(True)

        text_layout.addWidget(title_lbl)
        text_layout.addWidget(msg_lbl)

        close_btn = QPushButton("×")
        close_btn.setObjectName("toastCloseBtn")
        close_btn.setToolTip("Dismiss notification")
        close_btn.setFixedSize(20, 20)
        close_btn.clicked.connect(self.close_toast)

        layout.addWidget(icon_lbl)
        layout.addLayout(text_layout, 1)
        layout.addWidget(close_btn)

        self.setFixedWidth(420)
        self.adjustSize()
        self._reposition()

        # [ANNOTATION] Configure single-shot timer to dismiss notification automatically.
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close_toast)
        self.timer.start(duration_ms)

        self.show()
        self.raise_()

    def _reposition(self) -> None:
        # [ANNOTATION] Position notification in the top right corner of the parent container.
        if self.parentWidget():
            pw = self.parentWidget().width() # type: ignore
            self.move(pw - self.width() - 24, 20)

    def close_toast(self) -> None:
        # [ANNOTATION] Safely close and cleanup toast object.
        print("[TOAST] Dismissing notification overlay...")
        self.timer.stop()
        self.hide()
        self.deleteLater()

# =====================================================================
# UI Presentational Widgets
# =====================================================================

class MetricCard(QFrame):
    """Presentational metric card with title, large numerical value, and unit."""
    def setup(self, title: str, value: str = "-", unit: str = "") -> None:
        # [ANNOTATION] Build metric display layout containing label, value, and unit strings.
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self.title = QLabel(title)
        self.title.setObjectName("metricTitle")

        val_layout = QHBoxLayout()
        val_layout.setSpacing(4)
        val_layout.setContentsMargins(0, 0, 0, 0)

        self.value = QLabel(value)
        self.value.setObjectName("metricValue")

        self.unit = QLabel(unit)
        self.unit.setObjectName("metricUnit")
        self.unit.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)

        val_layout.addWidget(self.value)
        val_layout.addWidget(self.unit)
        val_layout.addStretch()

        layout.addWidget(self.title)
        layout.addLayout(val_layout)

    def set_value(self, value: Any, unit: str = "") -> None:
        # [ANNOTATION] Update metric card contents dynamically.
        self.value.setText(str(value))
        self.unit.setText(unit)

class StageWidget(QFrame):
    """Interactive pipeline stage indicator showing stage number, name, and status state."""
    def setup(self, number: str, title: str) -> None:
        # [ANNOTATION] Initialize sidebar stage tracker component.
        self.setObjectName("stage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        self.number = QLabel(number)
        self.number.setObjectName("stageNumber")

        self.icon_badge = QLabel("○")
        self.icon_badge.setObjectName("stageIcon")

        top_row.addWidget(self.number)
        top_row.addStretch()
        top_row.addWidget(self.icon_badge)

        self.title = QLabel(title)
        self.title.setObjectName("stageTitle")
        self.title.setWordWrap(True)

        self.state = QLabel("READY")
        self.state.setObjectName("stageState")

        layout.addLayout(top_row)
        layout.addWidget(self.title)
        layout.addWidget(self.state)

    def set_state(self, state: str) -> None:
        # [ANNOTATION] Update status badge, text, and dynamic stylesheet properties.
        self.state.setText(state)
        state_upper = state.upper()
        if "DONE" in state_upper or "COMPLETE" in state_upper:
            self.setProperty("state", "done")
            self.icon_badge.setText("✓")
        elif "ERROR" in state_upper or "FAIL" in state_upper:
            self.setProperty("state", "error")
            self.icon_badge.setText("✕")
        elif "RUNNING" in state_upper or "ACTIVE" in state_upper:
            self.setProperty("state", "running")
            self.icon_badge.setText("▶")
        elif "WARN" in state_upper:
            self.setProperty("state", "warning")
            self.icon_badge.setText("⚠")
        else:
            self.setProperty("state", "ready")
            self.icon_badge.setText("○")

        self.style().unpolish(self)
        self.style().polish(self)

# =====================================================================
# Background Worker Threads
# =====================================================================

class RealPipelineWorker(QThread):
    """Worker thread executing the real Chandrayaan-2 correspondence pipeline."""

    # [ANNOTATION] Define custom Qt Signals for async progress and result communication back to GUI main thread.
    sig_stage = Signal(int, str, str)
    sig_progress = Signal(int)
    sig_success = Signal(object, object, object)
    sig_error = Signal(int, str, str)

    def __init__(
        self,
        reference_raw: np.ndarray,
        source_raw: np.ndarray,
        preprocess_mode: str,
        config: PipelineConfig,
    ) -> None:
        super().__init__()
        print("[WORKER_REAL] Instantiating RealPipelineWorker thread...")
        self.reference_raw = reference_raw
        self.source_raw = source_raw
        self.preprocess_mode = preprocess_mode
        self.config = config

    def run(self) -> None:
        # [ANNOTATION] Async thread entry point executing the full 7-stage registration pipeline.
        stage_idx = 0
        try:
            print("\n" + "=" * 60)
            print("SIH26166 -- REAL CHANDRAYAAN-2 WORKFLOW (THREAD STARTED)")
            print("=" * 60)

            # [1/7] Ingestion / Input Validation
            stage_idx = 0
            self.sig_stage.emit(0, REAL_STAGE_NAMES[0], "RUNNING")
            self.sig_progress.emit(10)
            print(
                f"[1/7] Input imagery validated.\n"
                f"      Reference raw shape: {self.reference_raw.shape}\n"
                f"      Source raw shape:    {self.source_raw.shape}"
            )
            self.sig_stage.emit(0, REAL_STAGE_NAMES[0], "DONE")

            # [2/7] Preprocessing
            stage_idx = 1
            self.sig_stage.emit(1, REAL_STAGE_NAMES[1], "RUNNING")
            self.sig_progress.emit(25)
            print(f"[2/7] Preprocessing with method: {self.preprocess_mode}...")
            local_contrast = "CLAHE" if "CLAHE" in self.preprocess_mode else "NONE"
            pre_cfg = PreprocessingConfig(local_contrast=local_contrast)
            ref_proc = preprocess_for_features(self.reference_raw, config=pre_cfg)
            src_proc = preprocess_for_features(self.source_raw, config=pre_cfg)
            print(f"      Normalized range: [{ref_proc.min()}, {ref_proc.max()}] uint8")
            self.sig_stage.emit(1, REAL_STAGE_NAMES[1], "DONE")

            # [3/7] Feature Detection
            stage_idx = 2
            self.sig_stage.emit(2, REAL_STAGE_NAMES[2], "RUNNING")
            self.sig_progress.emit(40)
            print(f"[3/7] Detecting SIFT features (max={self.config.feature_max_features})...")
            ref_feats = detect_features(ref_proc, max_features=self.config.feature_max_features)
            src_feats = detect_features(src_proc, max_features=self.config.feature_max_features)
            print(
                f"      Reference features: {ref_feats.count}\n"
                f"      Source features:    {src_feats.count}"
            )
            self.sig_stage.emit(2, REAL_STAGE_NAMES[2], "DONE")

            # [4/7] Feature Matching
            stage_idx = 3
            self.sig_stage.emit(3, REAL_STAGE_NAMES[3], "RUNNING")
            self.sig_progress.emit(55)
            print(f"[4/7] Matching features ({self.config.matching.strategy})...")
            matches = match_features(ref_feats, src_feats, config=self.config.matching)
            print(f"      Initial matches: {matches.count}")
            if matches.count < self.config.geometric.min_inliers:
                raise ValueError(
                    f"Insufficient matches for geometric verification: "
                    f"{matches.count} available, {self.config.geometric.min_inliers} required."
                )
            self.sig_stage.emit(3, REAL_STAGE_NAMES[3], "DONE")

            # [5/7] Geometric Verification
            stage_idx = 4
            self.sig_stage.emit(4, REAL_STAGE_NAMES[4], "RUNNING")
            self.sig_progress.emit(70)
            print(
                f"[5/7] Robust geometric verification (model={self.config.geometric.model}, "
                f"thresh={self.config.geometric.reprojection_threshold}px)..."
            )
            geom_result = verify_geometry(matches, config=self.config.geometric)
            if not geom_result.success:
                raise ValueError(
                    f"Geometric verification failed. Inliers: {geom_result.inlier_count} "
                    f"(required >= {self.config.geometric.min_inliers})"
                )
            print(
                f"      Inliers: {geom_result.inlier_count}\n"
                f"      Inlier ratio: {geom_result.inlier_ratio * 100:.2f}%"
            )
            self.sig_stage.emit(4, REAL_STAGE_NAMES[4], "DONE")

            # [6/7] Spatial Selection & Sub-pixel Refinement
            stage_idx = 5
            self.sig_stage.emit(5, REAL_STAGE_NAMES[5], "RUNNING")
            self.sig_progress.emit(85)
            print(
                f"[6/7] Spatial selection (8x8 grid) and sub-pixel refinement..."
            )
            verified_mask = geom_result.inlier_mask
            verified_corrs = CorrespondenceSet(
                reference_points=matches.reference_points[verified_mask],
                source_points=matches.source_points[verified_mask],
                scores=None if matches.scores is None else matches.scores[verified_mask],
            )
            spatial_result = select_spatially_distributed(
                verified_corrs,
                image_shape=ref_proc.shape[:2],
                config=self.config.spatial,
            )
            subpixel_result = refine_correspondences(
                ref_proc,
                src_proc,
                spatial_result.correspondences,
                config=self.config.subpixel,
            )
            print(
                f"      Selected points: {spatial_result.selected_count}\n"
                f"      Refined points:  {subpixel_result.refined_count}\n"
                f"      Occupied cells:  {spatial_result.occupied_cells}/{spatial_result.total_cells}"
            )
            self.sig_stage.emit(5, REAL_STAGE_NAMES[5], "DONE")

            # [7/7] Registration & Diagnostic Evaluation
            stage_idx = 6
            self.sig_stage.emit(6, REAL_STAGE_NAMES[6], "RUNNING")
            self.sig_progress.emit(95)
            print(f"[7/7] Image registration ({self.config.registration.model})...")
            reg_result = register_image(
                source_image=src_proc,
                reference_shape=ref_proc.shape[:2],
                transformation=geom_result.transformation, # type: ignore
                config=self.config.registration,
            )

            refined_corrs = subpixel_result.correspondences
            predicted_source = _apply_transformation(
                refined_corrs.reference_points,
                geom_result.transformation, # type: ignore
                geom_result.model,
            )
            residual_metrics = transformation_residual_metrics(
                refined_corrs,
                predicted_source,
            )
            coverage = spatial_coverage(
                refined_corrs,
                image_shape=ref_proc.shape[:2],
                grid_rows=self.config.spatial.grid_rows,
                grid_cols=self.config.spatial.grid_cols,
            )

            pipeline_result = PipelineResult(
                reference_features=ref_feats,
                source_features=src_feats,
                initial_matches=matches,
                geometric_verification=geom_result,
                spatial_selection=spatial_result,
                subpixel_refinement=subpixel_result,
                registration=reg_result,
                residual_metrics=residual_metrics,
                spatial_coverage=coverage,
            )

            self.sig_stage.emit(6, REAL_STAGE_NAMES[6], "DONE")
            self.sig_progress.emit(100)

            print("\n" + "=" * 60)
            print("RESULTS (REAL CHANDRAYAAN-2 DIAGNOSTIC)")
            print("=" * 60)
            print(f"Matches:          {matches.count}")
            print(f"Inliers:          {geom_result.inlier_count}")
            print(f"Inlier ratio:     {geom_result.inlier_ratio:.3f}")
            print(
                f"Spatial coverage: {coverage.coverage_ratio:.3f} "
                f"({coverage.occupied_cells}/{coverage.total_cells} cells)"
            )
            print(f"Residual RMSE:    {residual_metrics.rmse:.3f} px")
            print(
                "Note: Real Chandrayaan-2 data has no independent pixel-level "
                "ground truth. Metrics are diagnostic residuals."
            )
            print("=" * 60 + "\n")

            # [ANNOTATION] Emit success signal back to main thread containing pipeline output.
            self.sig_success.emit(pipeline_result, ref_proc, src_proc)

        except Exception as exc:
            trace = traceback.format_exc()
            print(f"\n[ERROR] Stage [{stage_idx + 1}/7] {REAL_STAGE_NAMES[stage_idx]} failed:\n{trace}")
            self.sig_stage.emit(stage_idx, REAL_STAGE_NAMES[stage_idx], "ERROR")
            self.sig_error.emit(stage_idx, f"Stage {stage_idx + 1}: {REAL_STAGE_NAMES[stage_idx]} failed", str(exc))

class ControlledPipelineWorker(QThread):
    """Worker thread executing the controlled validation benchmark."""

    # [ANNOTATION] Define standard Qt signals for benchmark reporting.
    sig_stage = Signal(int, str, str)
    sig_progress = Signal(int)
    sig_success = Signal(object, object, object)
    sig_error = Signal(int, str, str)

    def __init__(
        self,
        reference_image: np.ndarray,
        source_image: np.ndarray,
        ground_truth: np.ndarray,
        config: PipelineConfig,
        model: str,
    ) -> None:
        super().__init__()
        print("[WORKER_CTRL] Instantiating ControlledPipelineWorker thread...")
        self.reference_image = reference_image
        self.source_image = source_image
        self.ground_truth = ground_truth
        self.config = config
        self.model = model

    def run(self) -> None:
        # [ANNOTATION] Execute controlled synthetic benchmark evaluation against known ground-truth matrix.
        stage_idx = 0
        try:
            print("\n" + "=" * 60)
            print("SIH26166 -- CONTROLLED VALIDATION WORKFLOW (THREAD STARTED)")
            print("=" * 60)

            stage_idx = 0
            self.sig_stage.emit(0, "LOAD BENCHMARK", "RUNNING")
            self.sig_progress.emit(10)
            print(
                f"[1/7] Loading controlled benchmark data...\n"
                f"      Reference shape: {self.reference_image.shape}\n"
                f"      Source shape:    {self.source_image.shape}\n"
                f"      Model:           {self.model}"
            )
            self.sig_stage.emit(0, "LOAD BENCHMARK", "DONE")

            self.sig_progress.emit(25)
            self.sig_stage.emit(1, "EVALUATION", "RUNNING")

            result = run_controlled_benchmark(
                self.reference_image,
                self.source_image,
                self.ground_truth,
                config=self.config,
                model=self.model, # type: ignore
            )

            self.sig_stage.emit(1, "EVALUATION", "DONE")
            self.sig_progress.emit(100)

            m = result.metrics
            print("\n" + "=" * 60)
            print("RESULTS (CONTROLLED VALIDATION -- GROUND TRUTH EVALUATION)")
            print("=" * 60)
            print(f"Transformation RMSE:      {m.transformation_rmse:.6f} px")
            print(f"Max Transformation Error: {m.transformation_max_error:.6f} px")
            print(f"Matches:                  {m.matched_count}")
            print(f"Inliers:                  {m.inlier_count}")
            print(f"Inlier ratio:             {m.inlier_ratio:.3f}")
            print(
                f"Spatial coverage:         {m.spatial_coverage_ratio:.3f} "
                f"({m.occupied_cells}/{m.total_cells} cells)"
            )
            print(f"Residual RMSE:            {m.residual_rmse:.3f} px")
            print("=" * 60 + "\n")

            self.sig_success.emit(result, self.reference_image, self.source_image)

        except Exception as exc:
            trace = traceback.format_exc()
            print(f"\n[ERROR] Controlled validation failed:\n{trace}")
            self.sig_stage.emit(stage_idx, "ERROR", "ERROR")
            self.sig_error.emit(stage_idx, "Controlled Benchmark Error", str(exc))

# =====================================================================
# Cinematic Startup Sequence Screen
# =====================================================================

class OrbitXStartupScreen(QWidget):
    """Cinematic ORBIT X aerospace mission system boot sequence."""

    # [ANNOTATION] Signal emitted when startup animation sequence finishes.
    finished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[STARTUP_SCREEN] Initializing splash boot screen...")
        self.setObjectName("orbitXStartupScreen")
        self.setStyleSheet("background-color: #050505;")

        self._progress = 0.0
        self._is_completed = False

        # [ANNOTATION] Detect headless environments or reduced motion settings to bypass animation.
        self._is_fast_mode = (
            os.environ.get("QT_QPA_PLATFORM") == "offscreen"
            or os.environ.get("SIH_REDUCED_MOTION", "").lower() in {"1", "true", "yes"}
        )

        duration = 50 if self._is_fast_mode else 2200

        # [ANNOTATION] Configure transition animation curve.
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(duration)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.valueChanged.connect(self._on_anim_value)
        self._anim.finished.connect(self._on_anim_finished)

    def start(self) -> None:
        # [ANNOTATION] Trigger boot sequence animation.
        print("[STARTUP_SCREEN] Executing startup boot animation sequence...")
        if self._is_completed:
            self.finished.emit()
            return

        if self._is_fast_mode:
            print("[STARTUP_SCREEN] Fast mode detected. Skipping animation duration.")
            self._progress = 1.0
            self._is_completed = True
            self.update()
            self.finished.emit()
            return

        self._anim.start()

    def _on_anim_value(self, val: float) -> None:
        self._progress = float(val)
        self.update()

    def _on_anim_finished(self) -> None:
        print("[STARTUP_SCREEN] Startup boot animation sequence finished.")
        if not self._is_completed:
            self._is_completed = True
            self.finished.emit()

    def paintEvent(self, event) -> None:
        # [ANNOTATION] Render dynamic mission startup HUD elements, grids, arcs, and typography.
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            w, h = self.width(), self.height()
            p = self._progress
            pad = 24

            painter.fillRect(0, 0, w, h, QColor(5, 5, 5))

            fade_alpha = 1.0
            if p > 0.85:
                fade_alpha = max(0.0, 1.0 - (p - 0.85) / 0.15)

            # [ANNOTATION] Draw tactical grid lines.
            grid_alpha = int(min(1.0, p / 0.25) * 22 * fade_alpha)
            if grid_alpha > 0:
                painter.setPen(QPen(QColor(243, 183, 24, grid_alpha), 1.0, Qt.PenStyle.DotLine))
                step_x = max(60, w // 12)
                step_y = max(60, h // 8)
                for x in range(step_x, w, step_x): painter.drawLine(x, 0, x, h)
                for y in range(step_y, h, step_y): painter.drawLine(0, y, w, y)

            # [ANNOTATION] Draw corner reticle brackets.
            bracket_alpha = int(min(1.0, p / 0.3) * 180 * fade_alpha)
            if bracket_alpha > 0:
                b_pen = QPen(QColor(243, 183, 24, bracket_alpha), 1.8)
                painter.setPen(b_pen)
                b_len = min(24, int(w * 0.02))
                painter.drawLine(pad, pad, pad + b_len, pad)
                painter.drawLine(pad, pad, pad, pad + b_len)
                painter.drawLine(w - pad, pad, w - pad - b_len, pad)
                painter.drawLine(w - pad, pad, w - pad, pad + b_len)
                painter.drawLine(pad, h - pad, pad + b_len, h - pad)
                painter.drawLine(pad, h - pad, pad, h - pad - b_len)
                painter.drawLine(w - pad, h - pad, w - pad - b_len, h - pad)
                painter.drawLine(w - pad, h - pad, w - pad, h - pad - b_len)

            # [ANNOTATION] Draw telemetry status strings.
            telemetry_alpha = int(min(1.0, p / 0.2) * 160 * fade_alpha)
            if telemetry_alpha > 0:
                t_font = QFont("Consolas", 8)
                t_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
                painter.setFont(t_font)
                painter.setPen(QColor(160, 160, 160, telemetry_alpha))
                painter.drawText(pad + 10, pad + 18, "SYS.BOOT // ORBITAL_NAV_SYS_v2.6")
                painter.drawText(w - pad - 220, pad + 18, f"INIT_SEQ: {int(p * 100):03d}% // READY")

            # [ANNOTATION] Draw orbital radar sweep effect.
            if p > 0.15:
                sweep_p = min(1.0, (p - 0.15) / 0.55)
                arc_alpha = int(min(1.0, sweep_p / 0.3) * 200 * fade_alpha)

                arc_rect = QRectF(w * 0.1, h * 0.22, w * 0.8, h * 0.56)
                painter.setPen(QPen(QColor(243, 183, 24, int(arc_alpha * 0.35)), 1.2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(arc_rect)

                sweep_x = w * 0.1 + sweep_p * (w * 0.8)
                painter.setPen(QPen(QColor(243, 183, 24, arc_alpha), 2.0))
                painter.drawLine(QPointF(sweep_x, h * 0.22), QPointF(sweep_x, h * 0.78))

                pulse_rad = 4.0
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(255, 255, 255, arc_alpha))
                painter.drawEllipse(QPointF(sweep_x, h * 0.5), pulse_rad, pulse_rad)

            # [ANNOTATION] Reveal system titles.
            if p > 0.35:
                reveal_p = min(1.0, (p - 0.35) / 0.45)
                title_alpha = int(reveal_p * 255 * fade_alpha)

                font_size = max(28, int(min(w, h) * 0.055))
                title_font = QFont("Consolas", font_size, QFont.Weight.Bold)
                title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 12.0)
                painter.setFont(title_font)

                title_text = "ORBIT X"
                fm = painter.fontMetrics()
                t_rect = fm.boundingRect(title_text)

                tx = (w - t_rect.width()) / 2
                ty = h / 2 - 10

                clip_w = t_rect.width() * reveal_p

                painter.save()
                painter.setPen(QColor(255, 255, 255, title_alpha))
                painter.drawText(QRectF(tx - 40, ty - 40, t_rect.width() + 80, 60), Qt.AlignmentFlag.AlignCenter, title_text)

                line_y = ty + 28
                painter.setPen(QPen(QColor(243, 183, 24, int(title_alpha * 0.9)), 2.0))
                painter.drawLine(QPointF((w - clip_w) / 2, line_y), QPointF((w + clip_w) / 2, line_y))
                painter.restore()

                if reveal_p > 0.25:
                    sub_p = min(1.0, (reveal_p - 0.25) / 0.75)
                    sub_alpha = int(sub_p * 220 * fade_alpha)

                    sub_font1 = QFont("Consolas", max(9, int(min(w, h) * 0.013)), QFont.Weight.Bold)
                    sub_font1.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4.0)
                    painter.setFont(sub_font1)
                    painter.setPen(QColor(243, 183, 24, sub_alpha))
                    painter.drawText(QRectF(0, line_y + 16, w, 24), Qt.AlignmentFlag.AlignCenter, "CHANDRAYAAN-2 OPTICAL CORRESPONDENCE")

                    sub_font2 = QFont("Consolas", max(8, int(min(w, h) * 0.010)))
                    sub_font2.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.5)
                    painter.setFont(sub_font2)
                    painter.setPen(QColor(160, 160, 160, int(sub_alpha * 0.85)))
                    painter.drawText(QRectF(0, line_y + 42, w, 20), Qt.AlignmentFlag.AlignCenter, "SIH26166 • ISRO")
        
        finally:
            painter.end()

# =====================================================================
# Custom Workflow UI Foundations
# =====================================================================

class WorkflowPageWidget(QWidget):
    """A custom widget acting as the base for workflow pages, extending the cinematic mission visual language."""
    def paintEvent(self, event) -> None:
        # [ANNOTATION] Render dynamic background grid pattern and telemetry highlights.
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            w, h = self.width(), self.height()
            
            painter.fillRect(0, 0, w, h, QColor(5, 5, 5))
            
            painter.setPen(QPen(QColor(255, 255, 255, 6), 1, Qt.PenStyle.DotLine))
            step = max(60, w // 25)
            for x in range(step, w, step):
                painter.drawLine(x, 0, x, h)
            for y in range(step, h, step):
                painter.drawLine(0, y, w, y)
                
            painter.setPen(QPen(QColor(243, 183, 24, 15), 1.5))
            painter.drawEllipse(QPointF(w, 0), h * 0.8, h * 0.8)
            painter.drawEllipse(QPointF(w, 0), h * 0.82, h * 0.82)
            
            painter.setPen(QPen(QColor(160, 160, 160, 30), 1.5))
            bracket = 20
            pad = 20
            painter.drawLine(pad, h - pad, pad + bracket, h - pad)
            painter.drawLine(pad, h - pad, pad, h - pad - bracket)
            painter.drawLine(w - pad, pad, w - pad - bracket, pad)
            painter.drawLine(w - pad, pad, w - pad, pad + bracket)
        finally:
            painter.end()


class WorkflowBufferingOverlay(QWidget):
    """Full-screen modal spacecraft telemetry buffering transition."""
    NUM_SEGMENTS = 28

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[BUFFER_OVERLAY] Constructing modal buffering transition overlay...")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._progress = 0.0
        self._on_finished_callback = None
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(1000)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.Linear)
        self._anim.valueChanged.connect(self._on_anim_value)
        self._anim.finished.connect(self._on_anim_finished)
        self.hide()

    def start(self, callback, duration_ms: int = 1000) -> None:
        # [ANNOTATION] Display buffering modal during workflow transitions.
        print("[BUFFER_OVERLAY] Triggering page transition buffering overlay...")
        self._on_finished_callback = callback
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen" or os.environ.get("SIH_REDUCED_MOTION", "").lower() in {"1", "true", "yes"}:
            print("[BUFFER_OVERLAY] Offscreen/reduced motion active. Invoking callback immediately.")
            if self._on_finished_callback:
                cb = self._on_finished_callback
                self._on_finished_callback = None
                cb()
            return

        self._progress = 0.0
        self._anim.stop()
        self._anim.setDuration(max(100, duration_ms))
        if self.parentWidget() is not None:
            self.setGeometry(0, 0, self.parentWidget().width(), self.parentWidget().height()) # type: ignore
        self.show()
        self.raise_()
        self._anim.start()

    def stop(self) -> None:
        # [ANNOTATION] Stop buffering transition and hide widget.
        self._anim.stop()
        self.hide()
        self._on_finished_callback = None

    def _on_anim_value(self, val: float) -> None:
        self._progress = float(val)
        self.update()

    def _on_anim_finished(self) -> None:
        print("[BUFFER_OVERLAY] Buffering animation completed. Navigating to requested view...")
        self.hide()
        if self._on_finished_callback:
            cb = self._on_finished_callback
            self._on_finished_callback = None
            cb()

    def closeEvent(self, event) -> None:
        self.stop()
        super().closeEvent(event)

    def paintEvent(self, event) -> None:
        # [ANNOTATION] Draw segmented buffering bar overlay.
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            w = self.width()
            h = self.height()
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 230))
            box_w = min(540, max(360, int(w * 0.45)))
            box_h = 136
            bx = (w - box_w) // 2
            by = (h - box_h) // 2
            painter.setPen(QPen(QColor(255, 255, 255, 15), 1.0))
            painter.setBrush(QBrush(QColor(5, 5, 5, 240)))
            painter.drawRoundedRect(QRectF(bx, by, box_w, box_h), 6.0, 6.0)
            bracket_len = 16
            b_pen = QPen(QColor(80, 80, 80, 200), 1.8)
            painter.setPen(b_pen)
            painter.drawLine(bx, by, bx + bracket_len, by)
            painter.drawLine(bx, by, bx, by + bracket_len)
            painter.drawLine(bx + box_w, by, bx + box_w - bracket_len, by)
            painter.drawLine(bx + box_w, by, bx + box_w, by + bracket_len)
            painter.drawLine(bx, by + box_h, bx + bracket_len, by + box_h)
            painter.drawLine(bx, by + box_h, bx, by + box_h - bracket_len)
            painter.drawLine(bx + box_w, by + box_h, bx + box_w - bracket_len, by + box_h)
            painter.drawLine(bx + box_w, by + box_h, bx + box_w, by + box_h - bracket_len)
            title_font = QFont("Consolas", 11, QFont.Weight.Bold)
            title_font.setStyleHint(QFont.StyleHint.Monospace)
            title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0)
            painter.setFont(title_font)
            painter.setPen(QColor(255, 255, 255, 245))
            painter.drawText(QRectF(bx, by + 18, box_w, 24), Qt.AlignmentFlag.AlignCenter, "DATA BUFFERING IN PROGRESS")
            bar_w = box_w - 48
            bar_h = 16
            seg_gap = 3
            num_segs = self.NUM_SEGMENTS
            seg_w = (bar_w - (num_segs - 1) * seg_gap) / num_segs
            bar_x = bx + 24
            bar_y = by + 56
            active_count = int(round(self._progress * num_segs))
            for i in range(num_segs):
                sx = bar_x + i * (seg_w + seg_gap)
                srect = QRectF(sx, bar_y, seg_w, bar_h)
                if i < active_count:
                    painter.setPen(QPen(QColor(255, 230, 100, 230), 1.0))
                    painter.setBrush(QBrush(QColor(243, 183, 24, 240)))
                    painter.drawRoundedRect(srect, 2.0, 2.0)
                    if i == active_count - 1 and active_count > 0:
                        painter.setPen(QPen(QColor(243, 183, 24, 90), 3.0))
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawRoundedRect(srect.adjusted(-1.5, -1.5, 1.5, 1.5), 2.5, 2.5)
                else:
                    painter.setPen(QPen(QColor(60, 60, 60, 150), 1.0))
                    painter.setBrush(QBrush(QColor(10, 10, 10, 190)))
                    painter.drawRoundedRect(srect, 2.0, 2.0)
            sub_font = QFont("Consolas", 8)
            sub_font.setStyleHint(QFont.StyleHint.Monospace)
            sub_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.8)
            painter.setFont(sub_font)
            painter.setPen(QColor(100, 100, 100, 210))
            pct = int(self._progress * 100)
            painter.drawText(
                QRectF(bx, by + 90, box_w, 20),
                Qt.AlignmentFlag.AlignCenter,
                f"CH-2 PAYLOAD SUBSYSTEM SYNC // {pct:02d}%",
            )
        finally:
            painter.end()


class RevealWrapper(QWidget):
    """A lightweight animated wrapper for achieving cinematic scroll reveals."""

    def __init__(self, child: QWidget, direction: str = "left", offset: int = 40, delay: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.child = child
        self.child.setParent(self)
        self.direction = direction
        self.max_offset = offset
        self.delay = delay
        self.revealed = False
        self._val = 0.0

        self.setSizePolicy(self.child.sizePolicy())
        if self.child.minimumSize().isValid():
            self.setMinimumSize(self.child.minimumSize())
        if self.child.maximumSize().isValid():
            self.setMaximumSize(self.child.maximumSize())

        self.effect = QGraphicsOpacityEffect(self.child)
        self.effect.setOpacity(0.0)
        self.child.setGraphicsEffect(self.effect)

        # [ANNOTATION] Set reveal translation and opacity animation parameters.
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(700)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.valueChanged.connect(self._on_val)
        self.anim.finished.connect(self._on_anim_finished)

    def _on_val(self, val: Any) -> None:
        self._val = float(val)
        if self.child.graphicsEffect() == self.effect:
            self.effect.setOpacity(self._val)
        self._update_geometry()

    def _on_anim_finished(self) -> None:
        self.child.setGraphicsEffect(None) # type: ignore
        self._update_geometry()

    def _update_geometry(self) -> None:
        if not self.child:
            return
        inv = 1.0 - self._val
        shift = int(self.max_offset * inv)
        cx = cy = 0
        
        if self.direction == "left": cx = shift
        elif self.direction == "right": cx = -shift
        elif self.direction == "bottom": cy = shift
        elif self.direction == "top": cy = -shift

        self.child.setGeometry(cx, cy, self.width(), self.height())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_geometry()

    def sizeHint(self) -> Any:
        return self.child.sizeHint() if self.child else super().sizeHint()

    def minimumSizeHint(self) -> Any:
        return self.child.minimumSizeHint() if self.child else super().minimumSizeHint()

    def reveal(self) -> None:
        # [ANNOTATION] Trigger reveal transition when element scrolls into view.
        if not self.revealed:
            self.revealed = True
            if self.delay > 0:
                QTimer.singleShot(self.delay, self.anim.start)
            else:
                self.anim.start()


class WorkflowCard(QFrame):
    """Native QFrame-based interactive workflow card with cinematic hover effects."""
    
    def __init__(self, title: str, subtitle: str, bullets: list[str], action: str, accent_color: str, callback, parent=None):
        super().__init__(parent)
        print(f"[CARD] Initializing WorkflowCard: '{title}'...")
        self.accent = QColor(accent_color)
        self.callback = callback
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(320)
        
        self._hover_val = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        
        self._setup_ui(title, subtitle, bullets, action)
        
    def _setup_ui(self, title: str, subtitle: str, bullets: list[str], action: str):
        # [ANNOTATION] Populate workflow selection card components and bullet items.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.setSpacing(8)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #ffffff; font-size: 22px; font-weight: 800; letter-spacing: 1px;")
        
        lbl_sub = QLabel(subtitle)
        lbl_sub.setStyleSheet("color: #aaaaaa; font-size: 13px; margin-bottom: 12px;")
        lbl_sub.setWordWrap(True)
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_sub)
        
        for b in bullets:
            row = QHBoxLayout()
            row.setSpacing(10)
            icon = QLabel("⯈")
            icon.setStyleSheet(f"color: {self.accent.name()}; font-size: 10px;")
            text = QLabel(b)
            text.setStyleSheet("color: #cccccc; font-size: 12px;")
            text.setWordWrap(True)
            row.addWidget(icon)
            row.addWidget(text, 1)
            layout.addLayout(row)
            
        layout.addStretch()
        
        self.action_lbl = QLabel(f"▶   {action}")
        self.action_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_lbl.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold; letter-spacing: 1px;")
        self.action_lbl.setFixedHeight(44)
        layout.addWidget(self.action_lbl)
        
    def _on_anim(self, val: float):
        self._hover_val = float(val)
        self.update()
        
    def enterEvent(self, event):
        # [ANNOTATION] Start highlight transition when cursor enters card boundaries.
        self._anim.stop()
        self._anim.setStartValue(self._hover_val)
        self._anim.setEndValue(1.0)
        self._anim.start()
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        # [ANNOTATION] Reverse highlight transition when cursor leaves.
        self._anim.stop()
        self._anim.setStartValue(self._hover_val)
        self._anim.setEndValue(0.0)
        self._anim.start()
        super().leaveEvent(event)
        
    def mousePressEvent(self, event: QMouseEvent):
        # [ANNOTATION] Execute designated workflow callback when user clicks card.
        if event.button() == Qt.MouseButton.LeftButton:
            print(f"[CARD] User selected workflow card: {self.action_lbl.text()}")
            self.callback()
        super().mousePressEvent(event)
        
    def paintEvent(self, event):
        # [ANNOTATION] Draw dynamic card border and glow effects.
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            r = self.rect().adjusted(2, 2, -2, -2)
            
            bg_base = QColor(10, 10, 10)
            bg_hover = QColor(16, 16, 16)
            bg_c = QColor(
                int(bg_base.red() + (bg_hover.red() - bg_base.red()) * self._hover_val),
                int(bg_base.green() + (bg_hover.green() - bg_base.green()) * self._hover_val),
                int(bg_base.blue() + (bg_hover.blue() - bg_base.blue()) * self._hover_val)
            )
            
            border_base = QColor(40, 40, 40)
            b_r = border_base.red() + (self.accent.red() - border_base.red()) * self._hover_val
            b_g = border_base.green() + (self.accent.green() - border_base.green()) * self._hover_val
            b_b = border_base.blue() + (self.accent.blue() - border_base.blue()) * self._hover_val
            b_a = 255 if self._hover_val > 0 else border_base.alpha()
            
            if self._hover_val > 0.01:
                glow_color = QColor(self.accent.red(), self.accent.green(), self.accent.blue(), int(25 * self._hover_val))
                painter.setBrush(glow_color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(r, 12, 12)
            
            painter.setBrush(bg_c)
            painter.setPen(QPen(QColor(int(b_r), int(b_g), int(b_b), b_a), 1.5))
            painter.drawRoundedRect(r, 12, 12)
            
            btn_r = self.action_lbl.geometry()
            btn_r_c = 25 + (self.accent.red() * 0.35 - 25) * self._hover_val
            btn_g_c = 25 + (self.accent.green() * 0.35 - 25) * self._hover_val
            btn_b_c = 25 + (self.accent.blue() * 0.35 - 25) * self._hover_val
            
            painter.setBrush(QColor(int(btn_r_c), int(btn_g_c), int(btn_b_c)))
            
            btn_border_r = 45 + (self.accent.red() - 45) * self._hover_val
            btn_border_g = 45 + (self.accent.green() - 45) * self._hover_val
            btn_border_b = 45 + (self.accent.blue() - 45) * self._hover_val
            
            painter.setPen(QPen(QColor(int(btn_border_r), int(btn_border_g), int(btn_border_b)), 1.2))
            painter.drawRoundedRect(btn_r, 6, 6)
        finally:
            painter.end()


class MissionLandingPage(WorkflowPageWidget):
    """Native QWidget-based mission landing screen with interactive layout."""

    def __init__(self, real_callback, controlled_callback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[LANDING_PAGE] Constructing MissionLandingPage home screen...")
        self._real_callback = real_callback
        self._controlled_callback = controlled_callback
        
        self._click_sound = None
        self._init_audio()
        
        self.reveal_widgets: list[RevealWrapper] = []
        
        self._setup_ui()
        self._buffer_overlay = WorkflowBufferingOverlay(self)

    def _init_audio(self) -> None:
        # [ANNOTATION] Locate and initialize sound effect files.
        sound_path = Path(__file__).parent / "ui_click.wav"
        if not sound_path.is_file():
            sound_path = PROJECT_ROOT / "src" / "python" / "sih26166" / "app" / "ui_click.wav"
        if sound_path.is_file() and HAS_MULTIMEDIA:
            try:
                print(f"[AUDIO] Found UI click sound file at: {sound_path}")
                self._click_sound = QSoundEffect(self)
                self._click_sound.setSource(QUrl.fromLocalFile(str(sound_path.resolve())))
                self._click_sound.setVolume(0.40)
            except Exception as exc:
                print(f"[AUDIO] Failed to load sound effect: {exc}")
                self._click_sound = None

    def play_click_sound(self) -> None:
        # [ANNOTATION] Play UI sound effect if initialized.
        try:
            if self._click_sound is not None:
                self._click_sound.play()
        except Exception:
            pass

    def _add_reveal(self, widget: QWidget, direction: str = "left", offset: int = 40, delay: int = 0) -> RevealWrapper:
        # [ANNOTATION] Utility helper wrapping widgets inside animated reveal components.
        wrapper = RevealWrapper(widget, direction, offset, delay)
        self.reveal_widgets.append(wrapper)
        return wrapper

    def _setup_ui(self) -> None:
        # [ANNOTATION] Build header, workflow cards, and scientific pipeline footer items.
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("background-color: transparent;")
        
        self.scroll_area.viewport().setStyleSheet("background-color: transparent;")
        self.scroll_area.viewport().setAutoFillBackground(False)
        
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("background-color: transparent;")
        
        scroll_layout = QVBoxLayout(self.scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        center_h = QHBoxLayout()
        center_h.addStretch(1)
        
        container = QWidget()
        container.setMaximumWidth(1300)
        content = QVBoxLayout(container)
        content.setContentsMargins(60, 40, 60, 40)
        
        # HEADER SECTION
        eyebrow = QLabel("SMART INDIA HACKATHON 2026 • ISRO PROBLEM STATEMENT SIH26166")
        eyebrow.setStyleSheet("color: #737373; font-size: 11px; font-weight: 800; letter-spacing: 2px;")
        
        hero = QLabel("Chandrayaan-2 Optical Image Correspondence")
        hero.setStyleSheet("color: #ffffff; font-size: 34px; font-weight: 900; letter-spacing: 0.5px;")
        
        sub = QLabel("Multi-modal, Sun-angle and scale-invariant correspondence for\nChandrayaan-2 optical imagery (OHRC, TMC-2, IIRS)")
        sub.setStyleSheet("color: #a3a3a3; font-size: 15px; line-height: 1.4;")
        
        content.addWidget(self._add_reveal(eyebrow, "left", 40, 0))
        content.addWidget(self._add_reveal(hero, "left", 50, 80))
        content.addWidget(self._add_reveal(sub, "left", 60, 160))
        content.addSpacing(40)
        
        # WORKFLOW SECTION
        choose = QLabel("CHOOSE WORKFLOW")
        choose.setStyleSheet("color: #ffffff; font-size: 12px; font-weight: bold; letter-spacing: 2px;")
        content.addWidget(self._add_reveal(choose, "left", 40, 240))
        content.addSpacing(10)
        
        cards = QHBoxLayout()
        cards.setSpacing(30)
        
        self.real_card = WorkflowCard(
            title="Real Chandrayaan-2 Workflow",
            subtitle="Process actual mission data from Chandrayaan-2",
            bullets=[
                "PDS4 / TMC-2 stereo pair (NCA, NCF)",
                "Actual mission acquisition (2020-08-04)",
                "Full 7-stage correspondence & registration pipeline",
                "Observational diagnostics & spatial grid evaluation",
                "Graceful automatic demonstration imagery fallback"
            ],
            action="Open Mission Workflow",
            accent_color=COLOR_CYAN,
            callback=self._on_real_activate
        )
        self.real_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.ctrl_card = WorkflowCard(
            title="Controlled Validation Benchmark",
            subtitle="Evaluate with known geometric and photometric perturbations",
            bullets=[
                "Known geometric & photometric perturbations",
                "Ground-truth transformation error evaluation",
                "Sub-pixel accuracy & RMSE validation",
                "Repeatable scientific verification benchmark",
                "Quantitative performance reporting"
            ],
            action="Open Benchmark Workflow",
            accent_color="#b88ae6", 
            callback=self._on_ctrl_activate
        )
        self.ctrl_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        cards.addWidget(self._add_reveal(self.real_card, "left", 70, 320))
        cards.addWidget(self._add_reveal(self.ctrl_card, "right", 70, 400))
        content.addLayout(cards)
        content.addStretch(1)
        
        # SCIENTIFIC PIPELINE FOOTER
        pipe_strip = QHBoxLayout()
        pipe_lbl = QLabel("SCIENTIFIC PIPELINE & DATASETS READY\nFully functional analytical correspondence")
        pipe_lbl.setStyleSheet("color: #16a34a; font-size: 10px; font-weight: bold; text-transform: uppercase;")
        
        pipe_strip.addWidget(self._add_reveal(pipe_lbl, "left", 40, 0))
        pipe_strip.addStretch()
        
        stages = [
            ("SIFT", "Feature Detection"),
            ("Ratio/Mutual", "Feature Matching"),
            ("RANSAC", "Robust Verification"),
            ("Spatial 8x8", "Distribution Grid"),
            ("Sub-Pixel", "Quadratic Refinement"),
            ("Homography", "Image Registration")
        ]
        
        for i, (st, desc) in enumerate(stages):
            col = QVBoxLayout()
            col.setSpacing(2)
            t1 = QLabel(st)
            t1.setStyleSheet("color: #e5e5e5; font-size: 12px; font-weight: bold;")
            t1.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t2 = QLabel(desc)
            t2.setStyleSheet("color: #737373; font-size: 10px;")
            t2.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(t1)
            col.addWidget(t2)
            
            w = QWidget()
            w.setLayout(col)
            w.setStyleSheet("border-left: 1px solid #222; padding-left: 15px; margin-left: 5px;")
            pipe_strip.addWidget(self._add_reveal(w, "bottom", 30, (i + 1) * 60))
            
        pipe_container = QWidget()
        pipe_container.setLayout(pipe_strip)
        content.addWidget(pipe_container)
        
        center_h.addWidget(container, 8)
        center_h.addStretch(1)
        
        scroll_layout.addLayout(center_h)
        self.scroll_area.setWidget(self.scroll_widget)
        main_layout.addWidget(self.scroll_area)
        
        self.scroll_area.verticalScrollBar().valueChanged.connect(self._check_reveals)

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event) # type: ignore
        QTimer.singleShot(50, self._check_reveals)

    def _check_reveals(self) -> None:
        # [ANNOTATION] Check viewport scroll position to trigger animations for elements coming into view.
        scroll_y = self.scroll_area.verticalScrollBar().value()
        viewport_height = self.scroll_area.viewport().height()
        threshold = scroll_y + viewport_height - 30
        
        for rw in self.reveal_widgets:
            if not rw.revealed:
                pos = rw.mapTo(self.scroll_widget, QPoint(0, 0))
                if pos.y() < threshold:
                    rw.reveal()

    def _on_real_activate(self) -> None:
        # [ANNOTATION] Handle user selection of Real Chandrayaan-2 Workflow card.
        print("[LANDING_PAGE] Real Chandrayaan-2 workflow card activated.")
        self.play_click_sound()
        self._buffer_overlay.start(self._real_callback, duration_ms=1000)

    def _on_ctrl_activate(self) -> None:
        # [ANNOTATION] Handle user selection of Controlled Validation Benchmark card.
        print("[LANDING_PAGE] Controlled Validation Benchmark card activated.")
        self.play_click_sound()
        self._buffer_overlay.start(self._controlled_callback, duration_ms=1000)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._buffer_overlay.setGeometry(0, 0, self.width(), self.height())
        self._check_reveals()

# =====================================================================
# Main Application Window & Overlays
# =====================================================================

class MainWindow(QMainWindow):
    """Main workstation window with Home, Real-data, and Controlled-validation pages."""
    def setup(self) -> None:
        # [ANNOTATION] Initialize main application window title, bounds, variables, and child pages.
        print("[MAIN_WINDOW] Setting up main workstation window...")
        self.setWindowTitle("SIH26166 — Chandrayaan-2 Optical Correspondence Workstation")
        self.setMinimumSize(960, 560)

        self.reference_raw: np.ndarray | None = None
        self.source_raw: np.ndarray | None = None
        self.nca_product: TMC2Product | None = None
        self.ncf_product: TMC2Product | None = None
        self.nca_metadata: PDS4ImageMetadata | None = None
        self.ncf_metadata: PDS4ImageMetadata | None = None
        self.is_demo_fallback: bool = False
        self.last_pipeline_config: PipelineConfig | None = None

        self.pipeline_result: PipelineResult | None = None
        self.controlled_result: BenchmarkResult | None = None
        self.worker: QThread | None = None
        self._active_toast: ToastNotification | None = None

        # [ANNOTATION] Create stacked widget manager to switch between different views.
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.startup_screen = OrbitXStartupScreen()
        self.startup_screen.finished.connect(self._on_startup_finished)
        self.stack.addWidget(self.startup_screen)

        self.build_home()
        self.build_real_page()
        self.build_controlled_page()
        
        self.stack.currentChanged.connect(self._on_stack_changed)

        self.stack.setCurrentWidget(self.startup_screen)
        QTimer.singleShot(10, self.startup_screen.start)

    def _on_startup_finished(self) -> None:
        # [ANNOTATION] Transition from boot screen to home landing page upon startup completion.
        print("[MAIN_WINDOW] Startup sequence finished. Transitioning to main home view.")
        if self.stack.currentWidget() == self.startup_screen:
            self.show_home()

    def show_home(self) -> None:
        # [ANNOTATION] Display landing home page.
        print("[NAVIGATION] Returning to home screen view...")
        self.stack.setCurrentWidget(self.home_page)

    def _on_stack_changed(self, index: int) -> None:
        # [ANNOTATION] Trigger reveal checks when active page changes in QStackedWidget.
        print(f"[NAVIGATION] Active stacked page changed to index {index}.")
        if self.stack.currentWidget() == self.real_page:
            QTimer.singleShot(50, self._check_real_reveals)
        elif self.stack.currentWidget() == self.ctrl_page:
            QTimer.singleShot(50, self._check_ctrl_reveals)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._active_toast is not None:
            if shiboken6.isValid(self._active_toast):
                if self._active_toast.isVisible():
                    self._active_toast._reposition()
            else:
                self._active_toast = None

    def show_toast(self, title: str, message: str, level: str = "info", duration_ms: int = 4000) -> None:
        # [ANNOTATION] Display non-blocking toast overlay notification.
        if self._active_toast is not None:
            if shiboken6.isValid(self._active_toast):
                if self._active_toast.isVisible():
                    self._active_toast.close_toast()
            else:
                self._active_toast = None

        self._active_toast = ToastNotification(self, title=title, message=message, level=level, duration_ms=duration_ms)

    def build_home(self) -> None:
        # [ANNOTATION] Instantiate landing home page.
        self.home_page = MissionLandingPage(
            real_callback=self.switch_to_real,
            controlled_callback=self.switch_to_controlled
        )
        self.stack.addWidget(self.home_page)

    def _build_header(self, title: str, subtitle: str, meta: str) -> QHBoxLayout:
        # [ANNOTATION] Helper to create standardized workflow section headers.
        header = QHBoxLayout()
        back = QPushButton("← HOME")
        back.setObjectName("secondaryButton")
        back.clicked.connect(self.show_home)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        lbl_sih = QLabel(title)
        lbl_sih.setObjectName("headerMain")
        lbl_sub = QLabel(subtitle)
        lbl_sub.setObjectName("headerSub")
        lbl_meta = QLabel(meta)
        lbl_meta.setObjectName("headerMeta")

        title_box.addWidget(lbl_sih)
        title_box.addWidget(lbl_sub)
        title_box.addWidget(lbl_meta)

        header.addWidget(back)
        header.addSpacing(16)
        header.addLayout(title_box)
        header.addStretch()
        return header
        
    def _create_section_header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a3a3a3; font-size: 12px; font-weight: bold; letter-spacing: 1.5px;")
        return lbl

    def build_real_page(self) -> None:
        # [ANNOTATION] Construct page layout for the Real Chandrayaan-2 Workflow page.
        print("[MAIN_WINDOW] Building Real Chandrayaan-2 workflow page UI...")
        self.real_page = WorkflowPageWidget()
        root = QVBoxLayout(self.real_page)
        root.setContentsMargins(0, 0, 0, 0)

        self.real_scroll = QScrollArea()
        self.real_scroll.setWidgetResizable(True)
        self.real_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.real_scroll.setStyleSheet("background-color: transparent;")
        self.real_scroll.viewport().setStyleSheet("background-color: transparent;")

        self.real_scroll_widget = QWidget()
        self.real_scroll_widget.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(self.real_scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        center_h = QHBoxLayout()
        center_h.addStretch(1)
        container = QWidget()
        container.setMaximumWidth(1350)
        content = QVBoxLayout(container)
        content.setContentsMargins(40, 40, 40, 60)
        content.setSpacing(24)

        self.real_reveals = []
        def add_reveal(w, d="left", off=40, dl=0):
            rw = RevealWrapper(w, d, off, dl)
            self.real_reveals.append(rw)
            return rw

        # SECTION 01 — WORKFLOW / MISSION HEADER
        head_container = QWidget()
        head_layout = self._build_header("SIH26166", "REAL CHANDRAYAAN-2 OPTICAL CORRESPONDENCE", "OHRC / TMC-2 / IIRS • REAL DATA • MODEL: AFFINE/HOMOGRAPHY")
        head_container.setLayout(head_layout)
        content.addWidget(add_reveal(head_container, "top", 30, 0))

        div = QFrame(); div.setObjectName("divider")
        content.addWidget(add_reveal(div, "left", 30, 100))

        scroll_ind = QLabel("↓ SCROLL TO EXPLORE MISSION STAGES")
        scroll_ind.setStyleSheet("color: #737373; font-size: 10px; font-weight: 800; letter-spacing: 2px;")
        scroll_ind.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(add_reveal(scroll_ind, "top", 20, 200))
        content.addSpacing(10)

        # SECTION 02 — INPUT DATA & PARAMETERS
        content.addWidget(add_reveal(self._create_section_header("01 // MISSION PARAMETERS & INPUT DATA"), "left", 30, 0))

        controls = QFrame()
        controls.setObjectName("panel")
        panel_vbox = QVBoxLayout(controls)
        panel_vbox.setContentsMargins(16, 12, 16, 12)
        panel_vbox.setSpacing(10)

        field_grid = QGridLayout()
        field_grid.setContentsMargins(0, 0, 0, 0)
        field_grid.setHorizontalSpacing(14)
        field_grid.setVerticalSpacing(6)

        self.real_preprocess = QComboBox(); self.real_preprocess.addItems(["Percentile + CLAHE", "Percentile"])
        self.real_matching = QComboBox(); self.real_matching.addItems(["RATIO_MUTUAL", "RATIO", "MUTUAL"])
        self.real_geometry = QComboBox(); self.real_geometry.addItems(["HOMOGRAPHY", "AFFINE"])
        self.real_features = QSpinBox(); self.real_features.setRange(100, 5000); self.real_features.setValue(2000)
        self.real_ratio = QDoubleSpinBox(); self.real_ratio.setRange(0.50, 0.95); self.real_ratio.setValue(0.75); self.real_ratio.setSingleStep(0.05)
        self.real_max_matches = QSpinBox(); self.real_max_matches.setRange(50, 5000); self.real_max_matches.setValue(2000)
        self.real_threshold = QDoubleSpinBox(); self.real_threshold.setRange(0.5, 10.0); self.real_threshold.setValue(3.0); self.real_threshold.setSingleStep(0.5)
        self.real_max_points = QSpinBox(); self.real_max_points.setRange(20, 1000); self.real_max_points.setValue(500)
        self.real_radius = QSpinBox(); self.real_radius.setRange(1, 5); self.real_radius.setValue(2)

        fields = [
            ("PREPROCESS", self.real_preprocess), ("MATCHING", self.real_matching), 
            ("MODEL", self.real_geometry), ("MAX FEAT", self.real_features),
            ("RATIO", self.real_ratio), ("MAX MATCH", self.real_max_matches),
            ("THRESH", self.real_threshold), ("GRID PTS", self.real_max_points), ("SUBPX", self.real_radius)
        ]
        
        for i, (l, w) in enumerate(fields):
            row = i // 5
            col = (i % 5) * 2
            k = QLabel(l); k.setObjectName("fieldKey")
            field_grid.addWidget(k, row, col)
            field_grid.addWidget(w, row, col + 1)

        self.real_run = QPushButton("EXECUTE PIPELINE"); self.real_run.setObjectName("primaryButton"); self.real_run.clicked.connect(self.run_real)
        self.real_evidence = QPushButton("LOAD EVIDENCE"); self.real_evidence.setObjectName("secondaryButton"); self.real_evidence.clicked.connect(self.load_real_evidence)
        self.real_reload = QPushButton("RELOAD"); self.real_reload.setObjectName("secondaryButton"); self.real_reload.clicked.connect(self.load_tmc_pair)
        self.real_reset = QPushButton("RESET"); self.real_reset.setObjectName("secondaryButton"); self.real_reset.clicked.connect(self.reset_real)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(10)
        action_row.addWidget(self.real_run)
        action_row.addWidget(self.real_evidence)
        action_row.addWidget(self.real_reload)
        action_row.addWidget(self.real_reset)
        action_row.addStretch()

        panel_vbox.addLayout(field_grid)
        panel_vbox.addLayout(action_row)

        content.addWidget(add_reveal(controls, "bottom", 40, 300))

        self.real_progress = QProgressBar(); self.real_progress.setRange(0, 100); self.real_progress.setValue(0)
        content.addWidget(add_reveal(self.real_progress, "bottom", 20, 400))
        content.addSpacing(20)

        # SECTION 03 — CORRESPONDENCE PIPELINE
        content.addWidget(add_reveal(self._create_section_header("02 // CORRESPONDENCE PIPELINE & VISUAL ANALYSIS"), "left", 30, 0))

        workspace_container = QHBoxLayout()
        workspace_container.setSpacing(16)

        self.real_left_sidebar = QFrame()
        self.real_left_sidebar.setObjectName("panel")
        left_layout = QVBoxLayout(self.real_left_sidebar)
        left_layout.setSpacing(8)

        self.real_stages: list[StageWidget] = []
        for name in NEW_VISUAL_STAGE_NAMES:
            stage = StageWidget()
            stage.setup(name[:2], name[3:])
            self.real_stages.append(stage)
            left_layout.addWidget(stage)
        left_layout.addStretch()

        center = QVBoxLayout()
        center.setSpacing(12)
        self.real_mode_sel = ViewModeSelector()
        self.real_workspace = InteractiveImageWorkspace()
        self.real_workspace.setMinimumHeight(650)
        self.real_mode_sel.mode_changed.connect(self.real_workspace.set_mode)
        center.addWidget(self.real_mode_sel)
        center.addWidget(self.real_workspace, 1)

        workspace_container.addWidget(self.real_left_sidebar, 15)
        workspace_container.addLayout(center, 85)

        ws_widget = QWidget()
        ws_widget.setLayout(workspace_container)
        content.addWidget(add_reveal(ws_widget, "bottom", 60, 0))
        content.addSpacing(20)

        # SECTION 04 — CORRESPONDENCE RESULTS
        content.addWidget(add_reveal(self._create_section_header("03 // SCIENTIFIC METRICS & RESULTS"), "left", 30, 0))

        metrics_h = QHBoxLayout()
        metrics_h.setSpacing(12)

        self.real_m_matches = MetricCard()
        self.real_m_matches.setup("MATCHES")
        self.real_m_inliers = MetricCard()
        self.real_m_inliers.setup("INLIERS")
        self.real_m_ratio = MetricCard()
        self.real_m_ratio.setup("INLIER RATIO")
        self.real_m_rmse = MetricCard()
        self.real_m_rmse.setup("RESIDUAL RMSE", unit="px")
        self.real_m_cov = MetricCard()
        self.real_m_cov.setup("SPATIAL COVERAGE")

        for card in [self.real_m_matches, self.real_m_inliers, self.real_m_ratio, self.real_m_rmse, self.real_m_cov]:
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            metrics_h.addWidget(card)
            
        m_widget = QWidget()
        m_widget.setLayout(metrics_h)
        content.addWidget(add_reveal(m_widget, "bottom", 40, 0))
        content.addSpacing(20)

        # SECTION 05 — EVIDENCE / PROVENANCE
        content.addWidget(add_reveal(self._create_section_header("04 // EVIDENCE & PROVENANCE LOGS"), "left", 30, 0))

        self.real_metadata_text = QTextEdit()
        self.real_metadata_text.setReadOnly(True)
        self.real_metadata_text.setObjectName("matrixText")
        self.real_metadata_text.setMinimumHeight(200)
        content.addWidget(add_reveal(self.real_metadata_text, "bottom", 30, 0))

        center_h.addWidget(container, 1)
        center_h.addStretch(1)
        scroll_layout.addLayout(center_h)
        
        self.real_scroll.setWidget(self.real_scroll_widget)
        root.addWidget(self.real_scroll)
        
        self.real_scroll.verticalScrollBar().valueChanged.connect(self._check_real_reveals)
        self.stack.addWidget(self.real_page)

    def _check_real_reveals(self) -> None:
        if not hasattr(self, 'real_scroll') or not hasattr(self, 'real_scroll_widget'): return
        scroll_y = self.real_scroll.verticalScrollBar().value()
        viewport_height = self.real_scroll.viewport().height()
        threshold = scroll_y + viewport_height - 40
        
        for rw in getattr(self, 'real_reveals', []):
            if not rw.revealed:
                pos = rw.mapTo(self.real_scroll_widget, QPoint(0, 0))
                if pos.y() < threshold:
                    rw.reveal()

    def build_controlled_page(self) -> None:
        # [ANNOTATION] Construct page layout for Controlled Validation Benchmark workflow.
        print("[MAIN_WINDOW] Building Controlled Validation workflow page UI...")
        self.ctrl_page = WorkflowPageWidget()
        root = QVBoxLayout(self.ctrl_page)
        root.setContentsMargins(0, 0, 0, 0)

        self.ctrl_scroll = QScrollArea()
        self.ctrl_scroll.setWidgetResizable(True)
        self.ctrl_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.ctrl_scroll.setStyleSheet("background-color: transparent;")
        self.ctrl_scroll.viewport().setStyleSheet("background-color: transparent;")

        self.ctrl_scroll_widget = QWidget()
        self.ctrl_scroll_widget.setStyleSheet("background-color: transparent;")
        scroll_layout = QVBoxLayout(self.ctrl_scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        center_h = QHBoxLayout()
        center_h.addStretch(1)
        container = QWidget()
        container.setMaximumWidth(1350)
        content = QVBoxLayout(container)
        content.setContentsMargins(40, 40, 40, 60)
        content.setSpacing(24)

        self.ctrl_reveals = []
        def add_reveal(w, d="left", off=40, dl=0):
            rw = RevealWrapper(w, d, off, dl)
            self.ctrl_reveals.append(rw)
            return rw

        # SECTION 01 — WORKFLOW / MISSION HEADER
        head_container = QWidget()
        head_layout = self._build_header("VALIDATION", "CONTROLLED BENCHMARK GROUND TRUTH", "SYNTHETIC DEFORMATION • GROUND TRUTH EVALUATION")
        head_container.setLayout(head_layout)
        content.addWidget(add_reveal(head_container, "top", 30, 0))

        div = QFrame(); div.setObjectName("divider")
        content.addWidget(add_reveal(div, "left", 30, 100))
        
        scroll_ind = QLabel("↓ SCROLL TO EXPLORE BENCHMARK STAGES")
        scroll_ind.setStyleSheet("color: #737373; font-size: 10px; font-weight: 800; letter-spacing: 2px;")
        scroll_ind.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.addWidget(add_reveal(scroll_ind, "top", 20, 200))
        content.addSpacing(10)

        # SECTION 02 — INPUT DATA & PARAMETERS
        content.addWidget(add_reveal(self._create_section_header("01 // BENCHMARK PARAMETERS & INPUT DATA"), "left", 30, 0))

        controls = QFrame()
        controls.setObjectName("panel")
        panel_vbox = QVBoxLayout(controls)
        panel_vbox.setContentsMargins(16, 12, 16, 12)
        panel_vbox.setSpacing(10)

        field_grid = QGridLayout()
        field_grid.setContentsMargins(0, 0, 0, 0)
        field_grid.setHorizontalSpacing(14)
        field_grid.setVerticalSpacing(6)
        
        self.ctrl_model = QComboBox(); self.ctrl_model.addItems(["AFFINE", "HOMOGRAPHY"])
        self.ctrl_features = QSpinBox(); self.ctrl_features.setRange(100, 5000); self.ctrl_features.setValue(2000)
        self.ctrl_matching = QComboBox(); self.ctrl_matching.addItems(["RATIO_MUTUAL", "RATIO", "MUTUAL"])
        
        lbl_model = QLabel("MODEL"); lbl_model.setObjectName("fieldKey")
        lbl_feat = QLabel("MAX FEAT"); lbl_feat.setObjectName("fieldKey")
        lbl_match = QLabel("MATCH"); lbl_match.setObjectName("fieldKey")

        field_grid.addWidget(lbl_model, 0, 0)
        field_grid.addWidget(self.ctrl_model, 0, 1)
        field_grid.addWidget(lbl_feat, 0, 2)
        field_grid.addWidget(self.ctrl_features, 0, 3)
        field_grid.addWidget(lbl_match, 0, 4)
        field_grid.addWidget(self.ctrl_matching, 0, 5)

        self.ctrl_run = QPushButton("EVALUATE"); self.ctrl_run.setObjectName("primaryButtonControlled"); self.ctrl_run.clicked.connect(self.run_controlled)
        self.ctrl_reset = QPushButton("RESET"); self.ctrl_reset.setObjectName("secondaryButton"); self.ctrl_reset.clicked.connect(self.reset_controlled)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(10)
        action_row.addWidget(self.ctrl_run)
        action_row.addWidget(self.ctrl_reset)
        action_row.addStretch()

        panel_vbox.addLayout(field_grid)
        panel_vbox.addLayout(action_row)

        content.addWidget(add_reveal(controls, "bottom", 40, 300))

        self.ctrl_progress = QProgressBar(); self.ctrl_progress.setValue(0)
        content.addWidget(add_reveal(self.ctrl_progress, "bottom", 20, 400))
        content.addSpacing(20)

        # SECTION 03 — CORRESPONDENCE PIPELINE
        content.addWidget(add_reveal(self._create_section_header("02 // CORRESPONDENCE PIPELINE & VISUAL ANALYSIS"), "left", 30, 0))

        workspace_container = QHBoxLayout()
        workspace_container.setSpacing(16)

        self.ctrl_left_sidebar = QFrame()
        self.ctrl_left_sidebar.setObjectName("panel")
        left_layout = QVBoxLayout(self.ctrl_left_sidebar)
        left_layout.setSpacing(8)

        self.ctrl_stages: list[StageWidget] = []
        for name in NEW_VISUAL_STAGE_NAMES:
            stage = StageWidget()
            stage.setup(name[:2], name[3:])
            self.ctrl_stages.append(stage)
            left_layout.addWidget(stage)
        left_layout.addStretch()

        center = QVBoxLayout()
        center.setSpacing(12)
        self.ctrl_mode_sel = ViewModeSelector()
        self.ctrl_workspace = InteractiveImageWorkspace()
        self.ctrl_workspace.setMinimumHeight(650)
        self.ctrl_mode_sel.mode_changed.connect(self.ctrl_workspace.set_mode)
        center.addWidget(self.ctrl_mode_sel)
        center.addWidget(self.ctrl_workspace, 1)

        workspace_container.addWidget(self.ctrl_left_sidebar, 15)
        workspace_container.addLayout(center, 85)

        ws_widget = QWidget()
        ws_widget.setLayout(workspace_container)
        content.addWidget(add_reveal(ws_widget, "bottom", 60, 0))
        content.addSpacing(20)

        # SECTION 04 — CORRESPONDENCE RESULTS
        content.addWidget(add_reveal(self._create_section_header("03 // SCIENTIFIC METRICS & RESULTS"), "left", 30, 0))

        metrics_h = QHBoxLayout()
        metrics_h.setSpacing(12)

        self.ctrl_m_matches = MetricCard()
        self.ctrl_m_matches.setup("MATCHES")
        self.ctrl_m_inliers = MetricCard()
        self.ctrl_m_inliers.setup("INLIERS")
        self.ctrl_m_rmse = MetricCard()
        self.ctrl_m_rmse.setup("TRANSFORM RMSE", unit="px")
        self.ctrl_m_res = MetricCard()
        self.ctrl_m_res.setup("RESIDUAL RMSE", unit="px")
        self.ctrl_m_cov = MetricCard()
        self.ctrl_m_cov.setup("COVERAGE")
        
        for card in [self.ctrl_m_matches, self.ctrl_m_inliers, self.ctrl_m_rmse, self.ctrl_m_res, self.ctrl_m_cov]:
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            metrics_h.addWidget(card)

        m_widget = QWidget()
        m_widget.setLayout(metrics_h)
        content.addWidget(add_reveal(m_widget, "bottom", 40, 0))
        content.addSpacing(20)
        
        # SECTION 05 — EVIDENCE / GROUND TRUTH LOGS
        content.addWidget(add_reveal(self._create_section_header("04 // EVIDENCE & GROUND TRUTH LOGS"), "left", 30, 0))

        self.ctrl_gt = QTextEdit()
        self.ctrl_gt.setReadOnly(True)
        self.ctrl_gt.setObjectName("matrixText")
        self.ctrl_gt.setMinimumHeight(200)
        content.addWidget(add_reveal(self.ctrl_gt, "bottom", 30, 0))

        center_h.addWidget(container, 1)
        center_h.addStretch(1)
        scroll_layout.addLayout(center_h)
        
        self.ctrl_scroll.setWidget(self.ctrl_scroll_widget)
        root.addWidget(self.ctrl_scroll)
        
        self.ctrl_scroll.verticalScrollBar().valueChanged.connect(self._check_ctrl_reveals)
        self.stack.addWidget(self.ctrl_page)

    def _check_ctrl_reveals(self) -> None:
        if not hasattr(self, 'ctrl_scroll') or not hasattr(self, 'ctrl_scroll_widget'): return
        scroll_y = self.ctrl_scroll.verticalScrollBar().value()
        viewport_height = self.ctrl_scroll.viewport().height()
        threshold = scroll_y + viewport_height - 40
        
        for rw in getattr(self, 'ctrl_reveals', []):
            if not rw.revealed:
                pos = rw.mapTo(self.ctrl_scroll_widget, QPoint(0, 0))
                if pos.y() < threshold:
                    rw.reveal()

    def switch_to_real(self) -> None:
        # [ANNOTATION] Transition view stack to Real Chandrayaan-2 workflow page.
        print("[NAVIGATION] Switching view to Real Chandrayaan-2 page...")
        self.stack.setCurrentWidget(self.real_page)
        if self.reference_raw is None or self.source_raw is None:
            self.load_tmc_pair()

    def switch_to_controlled(self) -> None:
        # [ANNOTATION] Transition view stack to Controlled Benchmark workflow page.
        print("[NAVIGATION] Switching view to Controlled Benchmark page...")
        self.stack.setCurrentWidget(self.ctrl_page)
        self.load_controlled_inputs()

    def load_tmc_pair(self) -> None:
        # [ANNOTATION] Load actual Chandrayaan-2 TMC-2 image files or fallback samples.
        print("[DATA_LOADER] Executing TMC-2 image pair loading sequence...")
        try:
            nca_prod, ncf_prod = get_tmc2_20200804_pair(PROJECT_ROOT, validate_files=False)
            self.nca_product = nca_prod
            self.ncf_product = ncf_prod

            meta_str = "CHANDRAYAAN-2 DATA PROVENANCE\n" + "=" * 60 + "\n\n"
            meta_str += "Mission: Chandrayaan-2 (ISRO)\nInstrument: TMC-2\nPair: NCA ↔ NCF\n\n"

            if nca_prod.label_path.is_file():
                self.nca_metadata = read_pds4_label(nca_prod.label_path)
                meta_str += f"NCA ID: {self.nca_metadata.product_id}\n"
            if ncf_prod.label_path.is_file():
                self.ncf_metadata = read_pds4_label(ncf_prod.label_path)
                meta_str += f"NCF ID: {self.ncf_metadata.product_id}\n\n"

            meta_str += f"NCA Win: {NCA_WINDOW}\nNCF Win: {NCF_WINDOW}\n"
            self.real_metadata_text.setText(meta_str)

            has_raw_img = (nca_prod.image_path.is_file() and ncf_prod.image_path.is_file())

            if has_raw_img:
                print("[REAL DATA] Full TMC-2 .img products found. Reading windows with memory-mapping...")
                nca_full = read_image(nca_prod.image_path, nca_prod.image_spec, mmap=True, copy=False)
                ncf_full = read_image(ncf_prod.image_path, ncf_prod.image_spec, mmap=True, copy=False)
                x, y, w, h = NCA_WINDOW
                self.reference_raw = np.asarray(nca_full[y : y + h, x : x + w]).copy()
                x, y, w, h = NCF_WINDOW
                self.source_raw = np.asarray(ncf_full[y : y + h, x : x + w]).copy()
                self.is_demo_fallback = False
                self.show_toast("Mission Imagery Loaded", "Full Chandrayaan-2 TMC-2 .img products loaded.", "success")
            else:
                nca_sample = SAMPLES_DIR / "real_tmc2_nca_20200804_x576_y73576_2048.png"
                ncf_sample = SAMPLES_DIR / "real_tmc2_ncf_20200804_x476_y93676_2048.png"

                if not nca_sample.is_file() or not ncf_sample.is_file():
                    raise FileNotFoundError("Neither full TMC-2 .img products nor sample windows were found.")

                print("[REAL DATA] Using verified real Chandrayaan-2 TMC-2 sample windows (2048x2048)...")
                self.reference_raw = cv2.imread(str(nca_sample), cv2.IMREAD_GRAYSCALE)
                self.source_raw = cv2.imread(str(ncf_sample), cv2.IMREAD_GRAYSCALE)
                self.is_demo_fallback = True
                self.show_toast("Demonstration Mode", "Full raster unavailable. Using demonstration imagery.", "warning")

            if self.reference_raw is None or self.source_raw is None:
                raise ValueError("Failed to decode real Chandrayaan-2 image arrays.")

            self.real_workspace.set_data(
                array_to_pixmap(self.reference_raw),
                array_to_pixmap(self.source_raw),
                QPixmap(),
                QPixmap()
            )

        except Exception as exc:
            print(f"[ERROR] Failed to load TMC-2 image pair: {exc}")
            self.show_toast("Data Loading Error", str(exc), "error")

    def load_controlled_inputs(self) -> None:
        # [ANNOTATION] Load controlled benchmark sample dataset and ground-truth values.
        print("[DATA_LOADER] Loading controlled synthetic benchmark dataset...")
        try:
            ref_path = INTERIM_DIR / "controlled" / "benchmark_001" / "reference.png"
            src_path = INTERIM_DIR / "controlled" / "benchmark_001" / "source.png"
            gt_path = INTERIM_DIR / "controlled" / "benchmark_001" / "ground_truth.json"

            if ref_path.is_file() and src_path.is_file():
                self.ctrl_workspace.set_data(
                    QPixmap(str(ref_path)),
                    QPixmap(str(src_path)),
                    QPixmap(),
                    QPixmap()
                )

            if gt_path.is_file():
                data = json.loads(gt_path.read_text(encoding="utf-8"))
                matrix = data.get("reference_to_source_matrix", [])
                matrix_str = "\n".join("  ".join(f"{v: .8f}" for v in row) for row in matrix)
                gt_text = (
                    "CONTROLLED BENCHMARK GT\n"
                    + "=" * 30 + "\n"
                    f"Matrix:\n{matrix_str}\n\n"
                    f"Scale: {data.get('scale', 'N/A')}x\n"
                    f"Rotation: {data.get('rotation_deg', 'N/A')}°\n"
                    f"Trans X: {data.get('translation_x', 'N/A')} px\n"
                    f"Trans Y: {data.get('translation_y', 'N/A')} px\n"
                )
                self.ctrl_gt.setText(gt_text)

        except Exception as exc:
            print(f"[ERROR] Could not load controlled benchmark inputs: {exc}")

    def make_pipeline_config(self) -> PipelineConfig:
        # [ANNOTATION] Collect user-defined parameter settings from UI fields to form PipelineConfig object.
        print("[PIPELINE_CONFIG] Constructing PipelineConfig from current UI settings...")
        matching = MatchingConfig(
            strategy=self.real_matching.currentText(), # type: ignore
            ratio_threshold=self.real_ratio.value(),
            max_matches=self.real_max_matches.value(),
        )
        geometry = GeometricVerificationConfig(
            model=self.real_geometry.currentText(), # type: ignore
            reprojection_threshold=self.real_threshold.value(),
            confidence=0.995,
            max_iterations=5000,
            min_inliers=4,
        )
        spatial = SpatialSelectionConfig(
            grid_rows=8,
            grid_cols=8,
            max_points=self.real_max_points.value(),
        )
        subpixel = SubpixelRefinementConfig(
            method="QUADRATIC",
            window_radius=self.real_radius.value(),
            max_offset=0.75,
            minimum_curvature=1e-6,
        )
        registration = RegistrationConfig(
            model=self.real_geometry.currentText(), # type: ignore
            interpolation=1,
            border_mode=0,
            border_value=0.0,
        )
        return PipelineConfig(
            feature_max_features=self.real_features.value(),
            matching=matching,
            geometric=geometry,
            spatial=spatial,
            subpixel=subpixel,
            registration=registration,
        )

    def set_real_stage(self, index: int, state: str) -> None:
        # [ANNOTATION] Map numerical stage index to visual stage widget state.
        m = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 3}
        vis_idx = m.get(index, 0)
        for i, stage in enumerate(self.real_stages):
            if i < vis_idx:
                stage.set_state("DONE")
            elif i == vis_idx:
                stage.set_state(state)
            else:
                stage.set_state("READY")

    def _set_real_controls_enabled(self, enabled: bool) -> None:
        # [ANNOTATION] Enable/disable control parameters during execution.
        self.real_run.setEnabled(enabled)
        self.real_evidence.setEnabled(enabled)
        self.real_reload.setEnabled(enabled)
        self.real_reset.setEnabled(enabled)
        self.real_preprocess.setEnabled(enabled)
        self.real_matching.setEnabled(enabled)
        self.real_geometry.setEnabled(enabled)
        self.real_features.setEnabled(enabled)
        self.real_ratio.setEnabled(enabled)
        self.real_max_matches.setEnabled(enabled)
        self.real_threshold.setEnabled(enabled)
        self.real_max_points.setEnabled(enabled)
        self.real_radius.setEnabled(enabled)

    def reset_real(self) -> None:
        # [ANNOTATION] Clear results and reset real workflow controls.
        print("[ACTION] Resetting Real Chandrayaan-2 workspace...")
        self.pipeline_result = None
        for c in [self.real_m_matches, self.real_m_inliers, self.real_m_ratio, self.real_m_rmse, self.real_m_cov]: 
            c.set_value("-")
        for stage in self.real_stages: 
            stage.set_state("READY")
        self.real_progress.setValue(0)
        self.real_run.setText("EXECUTE PIPELINE")
        if self.reference_raw is not None and self.source_raw is not None:
            self.real_workspace.set_data(array_to_pixmap(self.reference_raw), array_to_pixmap(self.source_raw), QPixmap(), QPixmap())
        self.real_mode_sel.set_mode("REFERENCE")
        self.show_toast("Workspace Reset", "Real data correspondence workspace reset.", "info")

    def run_real(self) -> None:
        # [ANNOTATION] Trigger execution of background worker thread for Real Chandrayaan-2 workflow.
        print("[ACTION] Initiating Real Chandrayaan-2 correspondence pipeline...")
        if self.reference_raw is None or self.source_raw is None:
            self.load_tmc_pair()
            if self.reference_raw is None or self.source_raw is None:
                QMessageBox.critical(self, "Data Loading Error", "Could not load real Chandrayaan-2 TMC-2 image data.")
                return

        config = self.make_pipeline_config()
        self.last_pipeline_config = config
        self._set_real_controls_enabled(False)
        self.real_run.setText("PROCESSING PIPELINE...")
        self.real_progress.setValue(0)

        for stage in self.real_stages:
            stage.set_state("READY")

        # [ANNOTATION] Instantiate and start thread worker.
        self.worker = RealPipelineWorker(
            reference_raw=self.reference_raw,
            source_raw=self.source_raw,
            preprocess_mode=self.real_preprocess.currentText(),
            config=config,
        )

        self.worker.sig_stage.connect(self._on_real_stage_update)
        self.worker.sig_progress.connect(self.real_progress.setValue)
        self.worker.sig_success.connect(self._on_real_pipeline_success)
        self.worker.sig_error.connect(self._on_real_pipeline_error)
        self.worker.start()

    @Slot(int, str, str)
    def _on_real_stage_update(self, index: int, name: str, status: str) -> None:
        self.set_real_stage(index, status)

    @Slot(object, object, object)
    def _on_real_pipeline_success(self, result: PipelineResult, ref_norm: np.ndarray, src_norm: np.ndarray) -> None:
        # [ANNOTATION] Handle successful completion of Real Pipeline thread.
        print("[PIPELINE_CALLBACK] Real pipeline thread reported successful completion.")
        self.pipeline_result = result
        self._set_real_controls_enabled(True)
        self.real_run.setText("RUN AGAIN")
        self.real_stages[4].set_state("DONE")

        metrics = result.residual_metrics
        coverage = result.spatial_coverage
        verification = result.geometric_verification
        matches = len(result.initial_matches.reference_points)
        inliers = int(verification.inlier_count)

        self.real_m_matches.set_value(matches)
        self.real_m_inliers.set_value(inliers)
        self.real_m_ratio.set_value(f"{verification.inlier_ratio:.3f}")
        self.real_m_rmse.set_value(f"{metrics.rmse:.3f}")
        self.real_m_cov.set_value(f"{coverage.coverage_ratio:.3f}", f"({coverage.occupied_cells}/{coverage.total_cells})")

        registered = result.registration.registered_image
        diff_pix = make_difference_visual(ref_norm, registered)
        
        self.real_workspace.set_data(
            array_to_pixmap(ref_norm),
            array_to_pixmap(src_norm),
            array_to_pixmap(registered),
            diff_pix,
            result=result
        )
        self.real_mode_sel.set_mode("CORRESPONDENCE")

        cfg_text = (
            "EXECUTION RESULTS\n"
            f"Verified Inliers: {inliers} / {matches} ({verification.inlier_ratio * 100:.1f}%)\n"
            f"Residual RMSE: {metrics.rmse:.4f} px\n"
        )
        self.real_metadata_text.append(cfg_text)

        self.show_toast("Correspondence Complete", f"Verified {inliers} inliers with {coverage.occupied_cells} grid cells occupied.", "success")

    @Slot(int, str, str)
    def _on_real_pipeline_error(self, stage_idx: int, title: str, details: str) -> None:
        # [ANNOTATION] Handle failures during real pipeline execution.
        print(f"[PIPELINE_CALLBACK] Real pipeline reported error at stage index {stage_idx}: {details}")
        self._set_real_controls_enabled(True)
        self.real_run.setText("RETRY PIPELINE")
        self.real_progress.setValue(0)
        self.show_toast("Pipeline Error", f"{title}: {details}", "error")

    def load_real_evidence(self) -> None:
        # [ANNOTATION] Load precomputed benchmark panel evidence images.
        print("[ACTION] Loading precomputed TMC-2 benchmark evidence panel...")
        try:
            visual_dir = INTERIM_DIR / "tmc2_benchmark" / "visuals"
            panel = visual_dir / "SIH26166_TMC2_REAL_EVIDENCE_PANEL.png"
            metadata_file = visual_dir / "clahe_ratio_mutual_homography_metadata.json"

            if panel.is_file():
                self.real_workspace.set_data(QPixmap(str(panel)), QPixmap(), QPixmap(), QPixmap())
                self.real_mode_sel.set_mode("REFERENCE")

            if metadata_file.is_file():
                data = json.loads(metadata_file.read_text(encoding="utf-8"))
                matches = data.get("matches") or data.get("matched_count", "-")
                inliers = data.get("inliers") or data.get("inlier_count", "-")
                ratio = data.get("inlier_ratio")
                rmse = data.get("rmse") or data.get("residual_rmse")
                coverage = data.get("coverage") or data.get("spatial_coverage_ratio")

                self.real_m_matches.set_value(matches)
                self.real_m_inliers.set_value(inliers)
                if ratio is not None: self.real_m_ratio.set_value(f"{float(ratio):.3f}")
                if rmse is not None: self.real_m_rmse.set_value(f"{float(rmse):.3f}")
                if coverage is not None: self.real_m_cov.set_value(f"{float(coverage):.3f}")

            self.show_toast("Evidence Loaded", "Pre-computed TMC-2 benchmark evidence panel loaded.", "info")
        except Exception as exc:
            self.show_toast("Evidence Loading Error", str(exc), "error")

    def reset_controlled(self) -> None:
        # [ANNOTATION] Reset controlled benchmark metrics and controls.
        print("[ACTION] Resetting controlled benchmark page...")
        self.controlled_result = None
        for c in [self.ctrl_m_matches, self.ctrl_m_inliers, self.ctrl_m_rmse, self.ctrl_m_res, self.ctrl_m_cov]:
            c.set_value("-")
        for stage in self.ctrl_stages:
            stage.set_state("READY")
        self.ctrl_progress.setValue(0)
        self.ctrl_run.setText("EVALUATE")
        self.ctrl_mode_sel.set_mode("REFERENCE")
        self.load_controlled_inputs()
        self.show_toast("Benchmark Reset", "Controlled validation benchmark reset.", "info")

    def run_controlled(self) -> None:
        # [ANNOTATION] Launch controlled synthetic benchmark execution thread.
        print("[ACTION] Initiating controlled benchmark evaluation pipeline...")
        try:
            ref_path = INTERIM_DIR / "controlled" / "benchmark_001" / "reference.png"
            src_path = INTERIM_DIR / "controlled" / "benchmark_001" / "source.png"
            gt_path = INTERIM_DIR / "controlled" / "benchmark_001" / "ground_truth.json"

            if not ref_path.is_file() or not src_path.is_file() or not gt_path.is_file():
                raise FileNotFoundError("Controlled benchmark files missing.")

            reference = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            source = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            gt_data = json.loads(gt_path.read_text(encoding="utf-8"))

            matrix_raw = gt_data.get("reference_to_source_matrix")
            if matrix_raw is None:
                raise ValueError("Ground-truth matrix missing.")

            ground_truth = np.asarray(matrix_raw, dtype=np.float64)
            model = self.ctrl_model.currentText()

            config = PipelineConfig(
                feature_max_features=self.ctrl_features.value(),
                matching=MatchingConfig(
                    strategy=self.ctrl_matching.currentText(), # type: ignore
                    ratio_threshold=0.75,
                    max_matches=2000,
                ),
                geometric=GeometricVerificationConfig(
                    model=model, # type: ignore
                    reprojection_threshold=3.0,
                    confidence=0.995,
                    max_iterations=5000,
                    min_inliers=4,
                ),
                spatial=SpatialSelectionConfig(grid_rows=8, grid_cols=8, max_points=500),
                subpixel=SubpixelRefinementConfig(method="QUADRATIC", window_radius=2, max_offset=0.75, minimum_curvature=1e-6),
                registration=RegistrationConfig(model=model, interpolation=1, border_mode=0, border_value=0.0), # type: ignore
            )

            self.ctrl_run.setEnabled(False)
            self.ctrl_run.setText("RUNNING BENCHMARK...")
            self.ctrl_progress.setValue(0)

            for stage in self.ctrl_stages:
                stage.set_state("READY")

            self.worker = ControlledPipelineWorker(
                reference_image=reference, # type: ignore
                source_image=source, # type: ignore
                ground_truth=ground_truth,
                config=config,
                model=model,
            )

            self.worker.sig_stage.connect(self._on_ctrl_stage_update)
            self.worker.sig_progress.connect(self.ctrl_progress.setValue)
            self.worker.sig_success.connect(self._on_ctrl_pipeline_success)
            self.worker.sig_error.connect(self._on_ctrl_pipeline_error)
            self.worker.start()

        except Exception as exc:
            self.show_toast("Controlled Benchmark Error", str(exc), "error")

    @Slot(int, str, str)
    def _on_ctrl_stage_update(self, index: int, name: str, status: str) -> None:
        if index == 0:
            self.ctrl_stages[0].set_state(status)
        elif index == 1:
            for i in range(1, 4):
                self.ctrl_stages[i].set_state("DONE")
            self.ctrl_stages[4].set_state(status)

    @Slot(object, object, object)
    def _on_ctrl_pipeline_success(self, result: BenchmarkResult, ref: np.ndarray, src: np.ndarray) -> None:
        # [ANNOTATION] Handle completion of controlled benchmark thread.
        print("[PIPELINE_CALLBACK] Controlled pipeline thread reported success.")
        self.controlled_result = result
        self.ctrl_run.setEnabled(True)
        self.ctrl_run.setText("RUN AGAIN")

        m = result.metrics
        self.ctrl_m_matches.set_value(m.matched_count)
        self.ctrl_m_inliers.set_value(m.inlier_count)
        self.ctrl_m_rmse.set_value(f"{m.transformation_rmse:.4f}")
        self.ctrl_m_res.set_value(f"{m.residual_rmse:.3f}")
        self.ctrl_m_cov.set_value(f"{m.spatial_coverage_ratio:.3f}", f"({m.occupied_cells}/{m.total_cells})")

        registered = result.pipeline.registration.registered_image
        diff_pix = make_difference_visual(ref, registered)

        self.ctrl_workspace.set_data(
            array_to_pixmap(ref),
            array_to_pixmap(src),
            array_to_pixmap(registered),
            diff_pix,
            result=result.pipeline,
            benchmark=result
        )
        self.ctrl_mode_sel.set_mode("CORRESPONDENCE")

        gt_info = (
            f"\nRESULTS:\n"
            f"Transform RMSE: {m.transformation_rmse:.6f} px\n"
            f"Max Error: {m.transformation_max_error:.6f} px\n"
            f"Residual RMSE: {m.residual_rmse:.4f} px\n"
        )
        self.ctrl_gt.append(gt_info)

        self.show_toast("Benchmark Evaluated", f"RMSE: {m.transformation_rmse:.4f} px against GT.", "success")

    @Slot(int, str, str)
    def _on_ctrl_pipeline_error(self, stage_idx: int, title: str, details: str) -> None:
        # [ANNOTATION] Handle errors from controlled pipeline worker thread.
        print(f"[PIPELINE_CALLBACK] Controlled pipeline thread error: {details}")
        self.ctrl_run.setEnabled(True)
        self.ctrl_run.setText("RETRY BENCHMARK")
        self.ctrl_progress.setValue(0)
        self.show_toast("Controlled Benchmark Error", f"{title}: {details}", "error")


def apply_theme(app: QApplication) -> None:
    # [ANNOTATION] Apply dark theme styling to QApplication.
    print("[THEME] Applying global dark theme stylesheet...")
    app.setStyle("Fusion")
    app.setStyleSheet(f"""
        QMainWindow, QWidget#centralWidget, QStackedWidget {{ background-color: {COLOR_BG_MAIN}; color: {COLOR_TEXT_PRIMARY}; }}
        QWidget {{ background-color: {COLOR_BG_MAIN}; color: #e5e5e5; font-family: "Inter", "Helvetica Neue", sans-serif; }}
        QLabel#headerMain {{ color: #ffffff; font-size: 22px; font-weight: 900; letter-spacing: 2px; text-transform: uppercase; }}
        QLabel#headerSub {{ color: {COLOR_CYAN}; font-size: 12px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; }}
        QLabel#headerMeta {{ color: #888888; font-size: 10px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; }}
        QLabel#fieldKey {{ color: #888888; font-size: 10px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }}
        QFrame#divider {{ background-color: {COLOR_BORDER_DEFAULT}; max-height: 1px; border: none; margin-bottom: 12px; }}

        QFrame#panel {{ background-color: {COLOR_SURFACE_PANEL}; border: 1px solid {COLOR_BORDER_DEFAULT}; border-radius: 0px; }}
        QFrame#imageWorkspace {{ background-color: #000000; border: 1px solid {COLOR_BORDER_DEFAULT}; border-radius: 0px; }}

        QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: #000000; border: 1px solid {COLOR_BORDER_DEFAULT}; border-radius: 0px; color: #ffffff; padding: 4px 8px; min-height: 22px; font-size: 11px; font-weight: 600;
        }}
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border: 1px solid {COLOR_BORDER_HOVER}; }}
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {COLOR_BORDER_HOVER}; }}
        QComboBox::drop-down {{ border: none; width: 18px; }}
        QComboBox QAbstractItemView {{ background-color: #0a0a0a; border: 1px solid {COLOR_BORDER_DEFAULT}; selection-background-color: #1a1505; selection-color: {COLOR_BORDER_HOVER}; color: #ffffff; }}

        QPushButton#primaryButton {{
            background-color: #0a0a0a; border: 1px solid {COLOR_CYAN}; color: {COLOR_CYAN}; font-weight: 800; padding: 8px 16px; font-size: 11px; border-radius: 0px; letter-spacing: 1px; text-transform: uppercase;
        }}
        QPushButton#primaryButton:hover {{
            background-color: #0a0a0a; border: 1px solid {COLOR_BORDER_HOVER}; color: {COLOR_BORDER_HOVER};
        }}
        QPushButton#primaryButton:pressed {{
            background-color: {COLOR_CYAN_ACCENT}; border: 1px solid {COLOR_CYAN_ACCENT}; color: #000000;
        }}
        QPushButton#primaryButton:disabled {{
            background-color: {COLOR_DISABLED_BG}; border: 1px solid {COLOR_DISABLED_BORDER}; color: {COLOR_DISABLED};
        }}

        QPushButton#primaryButtonControlled {{
            background-color: {COLOR_CONTROLLED_BORDER}; border: 1px solid {COLOR_CONTROLLED}; color: #ffffff; font-weight: 800; padding: 8px 16px; font-size: 11px; border-radius: 0px; letter-spacing: 1px; text-transform: uppercase;
        }}
        QPushButton#primaryButtonControlled:hover {{
            background-color: #9333ea; border: 1px solid {COLOR_BORDER_HOVER}; color: {COLOR_BORDER_HOVER};
        }}
        QPushButton#primaryButtonControlled:pressed {{
            background-color: #581c87; border: 1px solid #d8b4fe; color: #ffffff;
        }}
        QPushButton#primaryButtonControlled:disabled {{
            background-color: {COLOR_DISABLED_BG}; border: 1px solid {COLOR_DISABLED_BORDER}; color: {COLOR_DISABLED};
        }}

        QPushButton#secondaryButton {{
            background-color: #0a0a0a; border: 1px solid {COLOR_BORDER_DEFAULT}; color: #aaaaaa; padding: 8px 16px; font-size: 11px; font-weight: 800; border-radius: 0px; letter-spacing: 1px; text-transform: uppercase;
        }}
        QPushButton#secondaryButton:hover {{
            background-color: {COLOR_BORDER_DEFAULT}; border: 1px solid #ffffff; color: #ffffff;
        }}
        QPushButton#secondaryButton:pressed {{
            background-color: #ffffff; border: 1px solid #ffffff; color: #000000;
        }}
        QPushButton#secondaryButton:disabled {{
            background-color: #0a0a0a; border: 1px solid {COLOR_DISABLED_BORDER}; color: {COLOR_DISABLED};
        }}

        QPushButton#modeButton {{
            background-color: #050505; border: 1px solid transparent; border-bottom: 1px solid {COLOR_BORDER_DEFAULT}; color: #777777; padding: 8px 12px; font-size: 10px; font-weight: 800; border-radius: 0px; letter-spacing: 1px; text-transform: uppercase;
        }}
        QPushButton#modeButton:checked {{ 
            background-color: #050505; border: 1px solid {COLOR_BORDER_DEFAULT}; border-bottom: 2px solid {COLOR_CYAN}; color: {COLOR_CYAN}; 
        }}
        QPushButton#modeButton:hover:!checked {{ 
            background-color: #111111; color: #ffffff; 
        }}

        QFrame#stage {{ background-color: transparent; border: 1px solid transparent; border-bottom: 1px solid {COLOR_BORDER_SUBTLE}; border-radius: 0px; }}
        QFrame#stage[state="done"] {{ border-left: 2px solid {COLOR_SUCCESS}; background-color: {COLOR_SUCCESS_BG}; }}
        QFrame#stage[state="running"] {{ border-left: 2px solid {COLOR_CYAN}; background-color: {COLOR_CYAN_BG}; }}
        QFrame#stage[state="error"] {{ border-left: 2px solid {COLOR_ERROR}; background-color: {COLOR_ERROR_BG}; }}

        QLabel#stageNumber {{ color: #555555; font-size: 12px; font-weight: 900; font-family: "Consolas", monospace; }}
        QFrame#stage[state="done"] QLabel#stageNumber {{ color: {COLOR_SUCCESS}; }}
        QFrame#stage[state="running"] QLabel#stageNumber {{ color: {COLOR_CYAN}; }}

        QLabel#stageTitle {{ color: #dddddd; font-size: 11px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }}
        QLabel#stageState {{ color: #666666; font-size: 9px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; }}
        QFrame#stage[state="running"] QLabel#stageState {{ color: {COLOR_CYAN_TEXT}; }}

        QFrame#metricCard {{ background-color: #050505; border: 1px solid {COLOR_BORDER_DEFAULT}; border-radius: 0px; padding: 4px; }}
        QLabel#metricTitle {{ color: #aaaaaa; font-size: 10px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }}
        QLabel#metricValue {{ color: #ffffff; font-size: 26px; font-weight: 900; font-family: "Consolas", monospace; }}
        QLabel#metricUnit {{ color: #777777; font-size: 11px; padding-bottom: 4px; font-weight: 700; }}

        QProgressBar {{ background-color: #111111; border: 1px solid {COLOR_BORDER_DEFAULT}; height: 4px; text-align: center; }}
        QProgressBar::chunk {{ background-color: {COLOR_CYAN}; }}

        QLabel#matrixText, QTextEdit#matrixText {{
            background-color: #050505; border: 1px solid {COLOR_BORDER_DEFAULT}; color: #dddddd; font-family: "Consolas", monospace; font-size: 10px; line-height: 1.5; padding: 8px;
        }}
    """)
    

def main() -> int:
    # [ANNOTATION] Application main execution loop function.
    print("[APPLICATION_START] Launching QApplication instance...")
    app = QApplication(sys.argv)
    app.setApplicationName("SIH26166")
    apply_theme(app)

    # [ANNOTATION] Instantiate MainWindow and run setup configuration.
    window = MainWindow()
    window.setup()

    # [ANNOTATION] Query monitor geometry and position window centered on screen.
    screen = app.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        work_w = max(960, available.width() - 24)
        work_h = max(560, available.height() - 48)
        target_w = min(1600, work_w)
        target_h = min(900, work_h)
        x = available.x() + max(0, (available.width() - target_w) // 2)
        y = available.y() + max(0, (available.height() - target_h) // 2)
        window.setGeometry(x, y, target_w, target_h)
    
    print("[APPLICATION_START] Displaying MainWindow and entering event loop.")
    window.show()

    # [ANNOTATION] Start Qt event loop.
    return app.exec()


# [ANNOTATION] Standard Python entry check block.
if __name__ == "__main__":
    print("[MAIN_GUARD] Executing gui_7.py directly...")
    sys.exit(main())