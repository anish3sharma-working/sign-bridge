"""SignBridge's native PySide6 desktop interface."""
from __future__ import annotations
import json
import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QPushButton,
    QLabel, QFrame, QGridLayout, QComboBox, QLineEdit, QTextEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QProgressBar, QGroupBox, QFormLayout, QHeaderView, QAbstractItemView,
    QScrollArea, QSizePolicy
)
from config.settings import Settings, app_data_dir
from database.db import Database
from recognition.static_model import StaticClassifier
from services.speech_service import SpeechService, normalize_transcript
from training.train_static import train_static
from vision.camera_manager import CameraWorker

LOG = logging.getLogger("signbridge")

DARK_STYLE = """
QWidget { background:#111827; color:#e5e7eb; font-family:'Segoe UI'; font-size:10pt; }
QFrame#sidebar { background:#0b1220; border-right:1px solid #263244; }
QLabel#brand { color:#ffffff; font-size:19pt; font-weight:700; padding:12px 4px; }
QLabel#muted { color:#9ca3af; }
QLabel#pageTitle { color:#f9fafb; font-size:21pt; font-weight:700; }
QLabel#sectionTitle { color:#f9fafb; font-size:13pt; font-weight:600; }
QFrame#card, QGroupBox { background:#182235; border:1px solid #29364a; border-radius:12px; }
QGroupBox { margin-top:12px; padding:16px 12px 12px 12px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#dbeafe; }
QPushButton { background:#253247; color:#f3f4f6; border:1px solid #3a4a61; border-radius:8px; padding:9px 13px; }
QPushButton:hover { background:#34445d; }
QPushButton:disabled { color:#718096; background:#1a2434; }
QPushButton#primary { background:#2563eb; border-color:#2563eb; color:white; font-weight:600; }
QPushButton#primary:hover { background:#1d4ed8; }
QPushButton#danger { background:#7f1d1d; border-color:#991b1b; }
QPushButton#nav { text-align:left; background:transparent; border:0; padding:11px 14px; color:#bdc7d6; }
QPushButton#nav:checked { background:#1d2c43; color:#ffffff; border-left:3px solid #60a5fa; }
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { background:#0f1724; border:1px solid #36445a; border-radius:7px; padding:8px; selection-background-color:#2563eb; }
QTextEdit { line-height:1.4; }
QTableWidget { background:#101827; alternate-background-color:#162133; gridline-color:#2b394c; border:1px solid #29364a; border-radius:8px; }
QHeaderView::section { background:#202d41; color:#dbeafe; padding:8px; border:0; }
QProgressBar { background:#0f1724; border:1px solid #36445a; border-radius:6px; height:12px; text-align:center; }
QProgressBar::chunk { background:#3b82f6; border-radius:6px; }
QScrollArea { border:0; }
"""
LIGHT_STYLE = """
QWidget { background:#f4f7fb; color:#182230; font-family:'Segoe UI'; font-size:10pt; }
QFrame#sidebar { background:#eaf0f8; border-right:1px solid #d0d9e6; }
QLabel#brand { color:#17365c; font-size:19pt; font-weight:700; padding:12px 4px; }
QLabel#muted { color:#526174; }
QLabel#pageTitle { color:#142239; font-size:21pt; font-weight:700; }
QLabel#sectionTitle { color:#142239; font-size:13pt; font-weight:600; }
QFrame#card, QGroupBox { background:#ffffff; border:1px solid #dce4ef; border-radius:12px; }
QGroupBox { margin-top:12px; padding:16px 12px 12px 12px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#243b5b; }
QPushButton { background:#e5ebf3; color:#1d2939; border:1px solid #cbd5e1; border-radius:8px; padding:9px 13px; }
QPushButton:hover { background:#dbe5f1; }
QPushButton#primary { background:#2563eb; border-color:#2563eb; color:white; font-weight:600; }
QPushButton#danger { background:#fee2e2; color:#991b1b; border-color:#fecaca; }
QPushButton#nav { text-align:left; background:transparent; border:0; padding:11px 14px; color:#34445b; }
QPushButton#nav:checked { background:#dbeafe; color:#163e72; border-left:3px solid #2563eb; }
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { background:white; border:1px solid #cbd5e1; border-radius:7px; padding:8px; selection-background-color:#93c5fd; }
QTableWidget { background:white; alternate-background-color:#f2f6fb; gridline-color:#dce4ef; border:1px solid #dce4ef; border-radius:8px; }
QHeaderView::section { background:#e8eef6; color:#243b5b; padding:8px; border:0; }
QProgressBar { background:#e5ebf3; border:1px solid #cbd5e1; border-radius:6px; height:12px; }
QProgressBar::chunk { background:#3b82f6; border-radius:6px; }
QScrollArea { border:0; }
"""


def card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def title_block(title: str, subtitle: str):
    box = QVBoxLayout()
    t = QLabel(title); t.setObjectName("pageTitle")
    s = QLabel(subtitle); s.setObjectName("muted"); s.setWordWrap(True)
    box.addWidget(t); box.addWidget(s)
    return box


class TrainingWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, database_path: Path, model_dir: Path, mirrored_input=False,
                 min_hand_confidence=0.45, swap_handedness=True):
        super().__init__()
        self.database_path = database_path
        self.model_dir = model_dir
        self.mirrored_input = bool(mirrored_input)
        self.min_hand_confidence = float(min_hand_confidence)
        self.swap_handedness = bool(swap_handedness)
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            result = train_static(self.database_path, self.model_dir, self.progress.emit,
                                  self.cancel_event, self.mirrored_input,
                                  self.min_hand_confidence, self.swap_handedness)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class SignBridgeWindow(QMainWindow):
    speech_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignBridge — Local ISL Gesture Communication")
        self.resize(1360, 860)
        self.setMinimumSize(1050, 700)
        self.settings = Settings()
        self.db = Database(app_data_dir() / "signbridge.sqlite3")
        self.data_root = app_data_dir()
        self.model: StaticClassifier | None = None
        self.camera: CameraWorker | None = None
        self.training_worker: TrainingWorker | None = None
        self.speech = SpeechService(self.settings.get("speech_rate"))
        self.current_prediction = None
        self.current_confidence = 0.0
        self.current_candidates = []
        self._build_ui()
        self._load_model()
        self._refresh_all()
        self.apply_theme(self.settings.get("theme"))
        self.speech_error.connect(self._show_speech_error)

    def _build_ui(self):
        root = QWidget(); outer = QHBoxLayout(root); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(218)
        nav_layout = QVBoxLayout(sidebar); nav_layout.setContentsMargins(15, 18, 15, 16)
        brand = QLabel("SignBridge"); brand.setObjectName("brand")
        tagline = QLabel("PRIVATE · LOCAL · ACCESSIBLE"); tagline.setObjectName("muted"); tagline.setStyleSheet("font-size:8pt; letter-spacing:1px")
        nav_layout.addWidget(brand); nav_layout.addWidget(tagline); nav_layout.addSpacing(24)
        self.pages = QStackedWidget()
        self.page_names = ["Dashboard", "Live recognition", "Dataset manager", "Model training", "History", "Settings & privacy"]
        self.nav_buttons = []
        for idx, name in enumerate(self.page_names):
            button = QPushButton(name); button.setObjectName("nav"); button.setCheckable(True)
            button.clicked.connect(lambda checked=False, i=idx: self.navigate(i))
            nav_layout.addWidget(button); self.nav_buttons.append(button)
        nav_layout.addStretch(1)
        local = QLabel("All camera processing and data storage stay on this device."); local.setObjectName("muted"); local.setWordWrap(True)
        nav_layout.addWidget(local)
        outer.addWidget(sidebar)
        self.pages.addWidget(self._dashboard_page())
        self.pages.addWidget(self._live_page())
        self.pages.addWidget(self._dataset_page())
        self.pages.addWidget(self._training_page())
        self.pages.addWidget(self._history_page())
        self.pages.addWidget(self._settings_page())
        outer.addWidget(self.pages, 1)
        self.setCentralWidget(root)
        self.navigate(0)

    def _scroll_page(self, content_layout):
        body = QWidget(); body.setLayout(content_layout)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(body)
        wrapper = QWidget(); layout = QVBoxLayout(wrapper); layout.setContentsMargins(28, 24, 28, 28); layout.addWidget(scroll)
        return wrapper

    def _dashboard_page(self):
        layout = QVBoxLayout(); layout.setSpacing(18)
        layout.addLayout(title_block("Good day. This is SignBridge.", "A private, local-first desktop workspace for collecting and recognizing a small, trained vocabulary of Indian Sign Language gestures."))
        banner = card(); b = QHBoxLayout(banner)
        self.dashboard_model = QLabel("Recognition setup required"); self.dashboard_model.setObjectName("sectionTitle")
        self.dashboard_model_detail = QLabel("No trained model is installed. SignBridge will not guess or show fabricated predictions."); self.dashboard_model_detail.setObjectName("muted"); self.dashboard_model_detail.setWordWrap(True)
        col = QVBoxLayout(); col.addWidget(self.dashboard_model); col.addWidget(self.dashboard_model_detail); b.addLayout(col, 1)
        start = QPushButton("Open live recognition"); start.setObjectName("primary"); start.clicked.connect(lambda: self.navigate(1)); b.addWidget(start)
        layout.addWidget(banner)
        grid = QGridLayout(); grid.setHorizontalSpacing(14); grid.setVerticalSpacing(14)
        self.dashboard_camera = QLabel("Camera starts only when you request it")
        self.dashboard_vocab = QLabel("0 verified model classes")
        self.dashboard_samples = QLabel("0 local samples")
        for pos, (heading, widget) in enumerate([("Camera", self.dashboard_camera), ("Model vocabulary", self.dashboard_vocab), ("Dataset", self.dashboard_samples)]):
            pane = card(); p = QVBoxLayout(pane); h = QLabel(heading); h.setObjectName("muted"); widget.setObjectName("sectionTitle"); p.addWidget(h); p.addWidget(widget); grid.addWidget(pane, 0, pos)
        layout.addLayout(grid)
        section = QLabel("Start with local data"); section.setObjectName("sectionTitle"); layout.addWidget(section)
        row = QHBoxLayout()
        for text, idx in [("Manage gesture classes & samples", 2), ("Train and evaluate a model", 3), ("Review settings & privacy", 5)]:
            btn = QPushButton(text); btn.clicked.connect(lambda checked=False, i=idx: self.navigate(i)); row.addWidget(btn)
        layout.addLayout(row)
        note = QLabel("Hand detector: up to two hands, labeled by the signer’s anatomical left/right side. Static pose models require fresh training after the bilateral feature update; they do not translate continuous ISL.")
        note.setWordWrap(True); note.setObjectName("muted"); layout.addWidget(note)
        layout.addStretch(1)
        return self._scroll_page(layout)

    def _live_page(self):
        layout = QVBoxLayout(); layout.setSpacing(14)
        layout.addLayout(title_block("Live recognition", "Start the webcam explicitly. Frames and landmark inference are processed locally and are never uploaded."))
        panels = QHBoxLayout(); panels.setSpacing(14)
        left = QGroupBox("Camera"); lv = QVBoxLayout(left)
        self.video_label = QLabel("Camera is off\n\nSelect Start camera when ready")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_label.setMinimumSize(480, 300); self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background:#080d16; border-radius:10px; color:#9ca3af")
        lv.addWidget(self.video_label, 1)
        controls = QHBoxLayout(); self.camera_index = QSpinBox(); self.camera_index.setRange(0, 20); self.camera_index.setValue(int(self.settings.get("camera_index")))
        controls.addWidget(QLabel("Camera")); controls.addWidget(self.camera_index)
        self.landmark_toggle = QCheckBox("Show landmarks"); self.landmark_toggle.setChecked(bool(self.settings.get("show_landmarks"))); controls.addWidget(self.landmark_toggle)
        lv.addLayout(controls)
        row = QHBoxLayout(); self.start_camera_btn = QPushButton("Start camera"); self.start_camera_btn.setObjectName("primary"); self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn = QPushButton("Stop camera"); self.stop_camera_btn.setEnabled(False); self.stop_camera_btn.clicked.connect(self.stop_camera)
        row.addWidget(self.start_camera_btn); row.addWidget(self.stop_camera_btn); lv.addLayout(row)
        self.camera_status = QLabel("Camera stopped"); self.camera_status.setObjectName("muted"); lv.addWidget(self.camera_status)
        self.performance = QLabel("FPS —  ·  inference —"); self.performance.setObjectName("muted"); lv.addWidget(self.performance)
        right = QVBoxLayout()
        live_status = card(); ls = QVBoxLayout(live_status)
        h = QLabel("Current sign"); h.setObjectName("muted"); self.prediction_label = QLabel("—"); self.prediction_label.setObjectName("pageTitle")
        self.confidence_label = QLabel("Confidence —"); self.confidence_label.setObjectName("muted")
        self.hand_detection_status = QLabel("Hands: not checked yet"); self.hand_detection_status.setWordWrap(True); self.hand_detection_status.setObjectName("muted")
        self.recognition_status = QLabel("No trained model loaded"); self.recognition_status.setWordWrap(True)
        self.candidates_label = QLabel("Candidate predictions appear here when a model is available."); self.candidates_label.setWordWrap(True); self.candidates_label.setObjectName("muted")
        for item in (h, self.prediction_label, self.confidence_label, self.hand_detection_status, self.recognition_status, self.candidates_label): ls.addWidget(item)
        right.addWidget(live_status)
        sample_box = QGroupBox("Collect a labeled webcam sample"); sv = QVBoxLayout(sample_box)
        self.capture_class = QComboBox(); self.recording_group = QLineEdit(f"session-{datetime.now():%Y%m%d-%H%M}"); self.participant_group = QLineEdit(); self.participant_group.setPlaceholderText("Optional participant code (avoid personal details)")
        form = QFormLayout(); form.addRow("Gesture class", self.capture_class); form.addRow("Recording group", self.recording_group); form.addRow("Participant group", self.participant_group); sv.addLayout(form)
        cap_btn = QPushButton("Save current frame as sample"); cap_btn.clicked.connect(self.capture_sample); sv.addWidget(cap_btn)
        right.addWidget(sample_box)
        panels.addWidget(left, 3); panels.addLayout(right, 2); layout.addLayout(panels)
        transcript = QGroupBox("Editable transcript — words are added only after your confirmation"); tv = QVBoxLayout(transcript)
        self.transcript_edit = QTextEdit(); self.transcript_edit.setPlaceholderText("Confirm a stable prediction to add it, or type/edit text here."); self.transcript_edit.setMinimumHeight(100); tv.addWidget(self.transcript_edit)
        actions = QHBoxLayout()
        for text, callback in [("Add prediction", self.add_prediction), ("Undo word", self.undo_word), ("Clear", self.clear_transcript), ("Copy", self.copy_transcript), ("Save", self.save_transcript), ("Speak", self.speak_transcript), ("Stop speech", self.speech.stop), ("Normalize spacing", self.normalize_transcript)]:
            btn = QPushButton(text); btn.clicked.connect(callback); actions.addWidget(btn)
        tv.addLayout(actions); layout.addWidget(transcript)
        return self._scroll_page(layout)

    def _dataset_page(self):
        layout = QVBoxLayout(); layout.setSpacing(14)
        layout.addLayout(title_block("Dataset manager", "Organize permitted local examples. Imported files remain where they are; new webcam captures are saved under your local SignBridge data folder."))
        classes = QGroupBox("Gesture classes"); cv = QHBoxLayout(classes)
        self.class_name = QLineEdit(); self.class_name.setPlaceholderText("e.g. a validated static gesture label")
        self.class_description = QLineEdit(); self.class_description.setPlaceholderText("Optional description")
        add = QPushButton("Add class"); add.setObjectName("primary"); add.clicked.connect(self.add_class)
        cv.addWidget(self.class_name, 2); cv.addWidget(self.class_description, 2); cv.addWidget(add)
        layout.addWidget(classes)
        import_box = QGroupBox("Import local media"); iv = QHBoxLayout(import_box)
        self.import_class = QComboBox(); self.import_group = QLineEdit(); self.import_group.setPlaceholderText("Recording group (default uses import batch)")
        self.import_participant = QLineEdit(); self.import_participant.setPlaceholderText("Optional participant code")
        import_btn = QPushButton("Choose images/videos…"); import_btn.clicked.connect(self.import_media)
        iv.addWidget(QLabel("Label")); iv.addWidget(self.import_class); iv.addWidget(self.import_group); iv.addWidget(self.import_participant); iv.addWidget(import_btn)
        layout.addWidget(import_box)
        self.sample_table = QTableWidget(0, 7); self.sample_table.setHorizontalHeaderLabels(["Label", "Type", "Recording group", "Participant", "Split", "Source", "Added"]); self.sample_table.setAlternatingRowColors(True); self.sample_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); self.sample_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.sample_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.sample_table.cellDoubleClicked.connect(self.preview_sample)
        layout.addWidget(self.sample_table, 1)
        row = QHBoxLayout(); refresh = QPushButton("Refresh"); refresh.clicked.connect(self.refresh_dataset); preview = QPushButton("Preview selected"); preview.clicked.connect(self.preview_sample); delete = QPushButton("Delete selected sample"); delete.setObjectName("danger"); delete.clicked.connect(self.delete_sample)
        delete_cls = QPushButton("Delete selected class and its metadata"); delete_cls.setObjectName("danger"); delete_cls.clicked.connect(self.delete_class)
        export = QPushButton("Export metadata JSON…"); export.clicked.connect(self.export_dataset_metadata)
        row.addWidget(refresh); row.addWidget(preview); row.addWidget(export); row.addWidget(delete); row.addWidget(delete_cls); layout.addLayout(row)
        info = QLabel("Static-model training uses an image directly or the midpoint frame of an imported video. Dynamic signing is not trained or recognized by this release. Keep recordings from the same capture together and provide participant codes when available."); info.setObjectName("muted"); info.setWordWrap(True); layout.addWidget(info)
        return self._scroll_page(layout)

    def _training_page(self):
        layout = QVBoxLayout(); layout.setSpacing(14)
        layout.addLayout(title_block("Model training", "Train a local Random Forest from real, labeled landmark samples. No built-in vocabulary or benchmark score is supplied."))
        overview = card(); ov = QVBoxLayout(overview)
        self.training_dataset_summary = QLabel("Dataset not inspected yet"); self.training_dataset_summary.setWordWrap(True); ov.addWidget(self.training_dataset_summary)
        warning = QLabel("Leakage guard: SignBridge keeps participant groups together when supplied; otherwise it keeps each recording group together. Each label needs at least three distinct groups and enough class coverage for independent train, validation, and test sets."); warning.setObjectName("muted"); warning.setWordWrap(True); ov.addWidget(warning)
        layout.addWidget(overview)
        row = QHBoxLayout(); self.train_btn = QPushButton("Start training"); self.train_btn.setObjectName("primary"); self.train_btn.clicked.connect(self.start_training); self.cancel_train_btn = QPushButton("Cancel training"); self.cancel_train_btn.setEnabled(False); self.cancel_train_btn.clicked.connect(self.cancel_training); row.addWidget(self.train_btn); row.addWidget(self.cancel_train_btn); row.addStretch(1); layout.addLayout(row)
        self.training_progress = QProgressBar(); self.training_progress.setRange(0, 0); self.training_progress.setVisible(False); layout.addWidget(self.training_progress)
        self.training_status = QLabel("Training has not started."); self.training_status.setWordWrap(True); layout.addWidget(self.training_status)
        self.training_metrics = QTextEdit(); self.training_metrics.setReadOnly(True); self.training_metrics.setPlaceholderText("Validation and held-out test metrics will appear here after a successful training run."); layout.addWidget(self.training_metrics, 1)
        layout.addStretch(1)
        return self._scroll_page(layout)

    def _history_page(self):
        layout = QVBoxLayout(); layout.setSpacing(14)
        layout.addLayout(title_block("Transcript history", "Saved transcripts are local. Camera footage is not recorded automatically."))
        self.history_table = QTableWidget(0, 2); self.history_table.setHorizontalHeaderLabels(["Saved", "Transcript"]); self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers); self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents); self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch); layout.addWidget(self.history_table, 1)
        row = QHBoxLayout(); load = QPushButton("Open selected"); load.clicked.connect(self.open_history); export = QPushButton("Export selected as text…"); export.clicked.connect(self.export_history); delete = QPushButton("Delete selected"); delete.setObjectName("danger"); delete.clicked.connect(self.delete_history); clear = QPushButton("Clear all history"); clear.setObjectName("danger"); clear.clicked.connect(self.clear_history)
        for btn in (load, export, delete, clear): row.addWidget(btn)
        layout.addLayout(row)
        return self._scroll_page(layout)

    def _settings_page(self):
        layout = QVBoxLayout(); layout.setSpacing(14)
        layout.addLayout(title_block("Settings & privacy", "Preferences and data are stored locally. No account, telemetry, network inference, or cloud service is used."))
        box = QGroupBox("Recognition and accessibility settings"); form = QFormLayout(box)
        self.set_camera = QSpinBox(); self.set_camera.setRange(0, 20); self.set_camera.setValue(int(self.settings.get("camera_index")))
        self.set_width = QComboBox(); self.set_width.addItems(["640", "800", "1280"]); self.set_width.setCurrentText(str(self.settings.get("camera_width")))
        self.set_height = QComboBox(); self.set_height.addItems(["480", "600", "720"]); self.set_height.setCurrentText(str(self.settings.get("camera_height")))
        self.set_threshold = QDoubleSpinBox(); self.set_threshold.setRange(0.1, 0.99); self.set_threshold.setSingleStep(0.05); self.set_threshold.setValue(float(self.settings.get("confidence_threshold")))
        self.set_stability = QSpinBox(); self.set_stability.setRange(1, 15); self.set_stability.setValue(int(self.settings.get("stability_frames")))
        self.set_inference_fps = QSpinBox(); self.set_inference_fps.setRange(5, 30); self.set_inference_fps.setValue(int(self.settings.get("inference_fps")))
        self.set_hand_confidence = QDoubleSpinBox(); self.set_hand_confidence.setRange(0.2, 0.9); self.set_hand_confidence.setSingleStep(0.05); self.set_hand_confidence.setValue(float(self.settings.get("min_hand_confidence")))
        self.set_mirrored_input = QCheckBox("Flip camera/media horizontally (selfie orientation)")
        self.set_mirrored_input.setChecked(bool(self.settings.get("mirrored_input")))
        self.set_mirrored_input.setToolTip("Flips pixels before detection and mirrors the live preview. Enabled by default for a selfie-style camera view. Training applies the same flip to imported and captured images. Changing it requires retraining the model.")
        self.set_swap_handedness = QCheckBox("Swap detected Left/Right labels")
        self.set_swap_handedness.setChecked(bool(self.settings.get("swap_handedness")))
        self.set_swap_handedness.setToolTip("Camera-specific correction. Leave enabled if the app currently calls your left hand Right (or vice versa). This affects training features too, so changing it requires retraining.")
        self.set_speech = QCheckBox("Enable local text-to-speech"); self.set_speech.setChecked(bool(self.settings.get("speech_enabled")))
        self.set_rate = QSpinBox(); self.set_rate.setRange(80, 300); self.set_rate.setValue(int(self.settings.get("speech_rate")))
        self.set_theme = QComboBox(); self.set_theme.addItems(["dark", "light"]); self.set_theme.setCurrentText(self.settings.get("theme"))
        form.addRow("Camera index", self.set_camera); form.addRow("Requested resolution", self._pair(self.set_width, self.set_height)); form.addRow("Maximum inference frequency (FPS)", self.set_inference_fps); form.addRow("Hand detector confidence", self.set_hand_confidence); form.addRow("Minimum model score (uncalibrated)", self.set_threshold); form.addRow("Stable frames before confirmation", self.set_stability); form.addRow("Image orientation", self.set_mirrored_input); form.addRow("Hand label correction", self.set_swap_handedness); form.addRow("Speech rate", self.set_rate); form.addRow("Theme", self.set_theme); form.addRow("Speech", self.set_speech)
        layout.addWidget(box)
        row = QHBoxLayout(); save = QPushButton("Save settings"); save.setObjectName("primary"); save.clicked.connect(self.save_settings); reset = QPushButton("Reset settings"); reset.clicked.connect(self.reset_settings); row.addWidget(save); row.addWidget(reset); row.addStretch(1); layout.addLayout(row)
        privacy = QGroupBox("Privacy & data handling"); pv = QVBoxLayout(privacy)
        privacy_text = QLabel(f"\n• Webcam frames are processed on this computer only.\n• Camera use begins only when you click Start camera and stops when you stop or close the app.\n• Images captured through the dataset workflow are saved locally; imported files are referenced in place.\n• Transcripts, settings, and model metadata are stored in the local application folder:\n  {self.data_root}\n• No analytics, accounts, external APIs, cloud inference, or automatic video recording.\n• Delete individual samples or transcripts from their sections. Deleting a sample entry does not delete an imported original file.\n• Only load model artifacts you trained yourself and trust; joblib files can execute code when loaded.")
        privacy_text.setWordWrap(True); pv.addWidget(privacy_text)
        open_folder = QPushButton("Open local data folder"); open_folder.clicked.connect(self.open_data_folder); pv.addWidget(open_folder)
        layout.addWidget(privacy); layout.addStretch(1)
        return self._scroll_page(layout)

    @staticmethod
    def _pair(left, right):
        row = QWidget(); l = QHBoxLayout(row); l.setContentsMargins(0, 0, 0, 0); l.addWidget(left); l.addWidget(QLabel("×")); l.addWidget(right); return row

    def apply_theme(self, theme):
        self.setStyleSheet(LIGHT_STYLE if theme == "light" else DARK_STYLE)

    def navigate(self, index: int):
        if hasattr(self, "pages") and index != 1 and self.camera and self.camera.isRunning():
            self.stop_camera()
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        if index == 2: self.refresh_dataset()
        if index == 3: self.refresh_training_summary()
        if index == 4: self.refresh_history()
        if index == 1: self.refresh_class_combos()

    def _load_model(self):
        record = self.db.latest_model()
        if not record or not Path(record["model_path"]).is_file():
            self.model = None
            return
        try:
            model_path = Path(record["model_path"]).resolve()
            trusted_dir = (self.data_root / "models").resolve()
            if not model_path.is_relative_to(trusted_dir):
                raise ValueError("The recorded model is outside the SignBridge local model folder.")
            self.model = StaticClassifier.load(model_path, float(self.settings.get("confidence_threshold")), bool(self.settings.get("mirrored_input")), bool(self.settings.get("swap_handedness")))
        except Exception as exc:
            self.model = None
            LOG.warning("Could not load saved model: %s", exc)

    def _refresh_all(self):
        self.refresh_dashboard(); self.refresh_class_combos(); self.refresh_dataset(); self.refresh_history(); self.refresh_training_summary()

    def refresh_dashboard(self):
        classes = self.db.list_classes(); samples = self.db.list_samples()
        self.dashboard_vocab.setText(f"{len(self.model.classes) if self.model else 0} verified model classes")
        self.dashboard_samples.setText(f"{len(samples)} local samples across {len(classes)} classes")
        self.dashboard_camera.setText("Stopped" if not (self.camera and self.camera.isRunning()) else "Active")
        if self.model:
            self.dashboard_model.setText("Trained model available")
            self.dashboard_model_detail.setText("Loaded local model classes: " + ", ".join(self.model.classes))
        else:
            self.dashboard_model.setText("Recognition setup required")
            self.dashboard_model_detail.setText("No compatible trained model is installed. Collect genuine samples and train; predictions stay disabled until then.")

    def refresh_class_combos(self):
        classes = self.db.list_classes()
        for combo in (getattr(self, "capture_class", None), getattr(self, "import_class", None)):
            if combo is None: continue
            current = combo.currentData()
            combo.clear()
            for cls in classes:
                combo.addItem(f"{cls['label']}  ·  {cls['sample_count']} samples", cls["id"])
            if current is not None:
                idx = combo.findData(current)
                if idx >= 0: combo.setCurrentIndex(idx)

    def refresh_dataset(self):
        if not hasattr(self, "sample_table"): return
        rows = self.db.list_samples(); self.sample_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [row["label"], row["sample_type"], row["recording_group"], row["participant_group"], row["dataset_split"], Path(row["source_path"]).name, row["created_at"]]
            for c, value in enumerate(vals):
                item = QTableWidgetItem(str(value)); item.setData(Qt.ItemDataRole.UserRole, row["id"]); item.setToolTip(row["source_path"] if c == 5 else str(value)); self.sample_table.setItem(r, c, item)
        self.refresh_class_combos(); self.refresh_dashboard()

    def add_class(self):
        try:
            self.db.add_class(self.class_name.text(), self.class_description.text())
            self.class_name.clear(); self.class_description.clear(); self.refresh_dataset()
        except Exception as exc: QMessageBox.warning(self, "Could not add class", str(exc))

    def delete_class(self):
        cls_id = self.import_class.currentData()
        if cls_id is None: return
        name = self.import_class.currentText()
        answer = QMessageBox.question(self, "Delete gesture class", f"Delete {name} and its dataset metadata? Webcam-captured frames in SignBridge storage will also be removed; imported original files will be left untouched.")
        if answer == QMessageBox.StandardButton.Yes:
            for sample in self.db.list_samples():
                if sample["class_id"] == cls_id: self._delete_captured_asset(sample["source_path"])
            self.db.delete_class(cls_id); self.refresh_dataset()

    def preview_sample(self, *_):
        row = self.sample_table.currentRow()
        if row < 0: return
        sample_id = self.sample_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        sample = next((x for x in self.db.list_samples() if x["id"] == sample_id), None)
        if not sample: return
        path = Path(sample["source_path"])
        cap = None
        try:
            import cv2
            if sample["sample_type"] == "video":
                cap = cv2.VideoCapture(str(path))
                ok, frame = cap.read()
                if not ok: frame = None
                note = "Previewing the first decodable video frame; video is not played or recorded."
            else:
                frame = cv2.imread(str(path))
                note = "Local image preview."
            if frame is None: raise ValueError("The selected file could not be decoded for preview.")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
            dialog = QDialog(self); dialog.setWindowTitle(f"Sample preview — {path.name}"); dialog.resize(900, 700)
            layout = QVBoxLayout(dialog); preview = QLabel(); preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setPixmap(QPixmap.fromImage(image).scaled(860, 560, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            layout.addWidget(preview); caption = QLabel(f"{note}\n{path}"); caption.setWordWrap(True); caption.setObjectName("muted"); layout.addWidget(caption)
            dialog.exec()
        except Exception as exc: QMessageBox.warning(self, "Sample preview unavailable", str(exc))
        finally:
            if cap is not None: cap.release()

    def export_dataset_metadata(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export dataset metadata", "SignBridge-dataset-metadata.json", "JSON files (*.json)")
        if not path: return
        payload = {"schema_version": 1, "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"), "classes": self.db.list_classes(), "samples": self.db.list_samples(), "note": "Metadata only; original media files are not copied."}
        try:
            Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            QMessageBox.information(self, "Metadata exported", "Dataset metadata was exported locally. Review participant codes and source paths before sharing the JSON file.")
        except OSError as exc: QMessageBox.warning(self, "Export failed", str(exc))

    def _delete_captured_asset(self, source_path):
        root = (self.data_root / "dataset" / "raw").resolve()
        path = Path(source_path).resolve()
        try:
            if path.is_relative_to(root) and path.is_file(): path.unlink()
        except OSError as exc: LOG.warning("Could not delete captured sample file %s: %s", path, exc)

    def import_media(self):
        if self.import_class.currentData() is None:
            QMessageBox.information(self, "Choose a label", "Create and select a gesture class first."); return
        files, _ = QFileDialog.getOpenFileNames(self, "Import local images or videos", str(Path.home()), "Media (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm)")
        if not files: return
        try: import cv2
        except ImportError:
            QMessageBox.critical(self, "OpenCV unavailable", "Install project requirements before importing media."); return
        label = self.import_class.currentText().split("  ·")[0]
        participant = self.import_participant.text().strip()
        successes, failures = 0, []
        batch_group = self.import_group.text().strip() or f"import-{datetime.now():%Y%m%d-%H%M%S}"
        for file in files:
            path = Path(file)
            typ = "video" if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"} else "image"
            if typ == "image":
                valid = cv2.imread(str(path)) is not None
            else:
                cap = cv2.VideoCapture(str(path)); valid = cap.isOpened(); cap.release()
            if not valid:
                failures.append(f"{path.name}: could not decode/open"); continue
            group = batch_group
            try:
                self.db.add_sample(label, path, typ, group, participant); successes += 1
            except Exception as exc: failures.append(f"{path.name}: {exc}")
        self.import_group.clear(); self.refresh_dataset()
        msg = f"Added {successes} valid file(s)."
        if failures: msg += "\n\nSkipped:\n" + "\n".join(failures[:10])
        QMessageBox.information(self, "Import results", msg)

    def delete_sample(self):
        row = self.sample_table.currentRow()
        if row < 0: return
        item = self.sample_table.item(row, 0); sample_id = item.data(Qt.ItemDataRole.UserRole)
        sample = next((x for x in self.db.list_samples() if x["id"] == sample_id), None)
        if QMessageBox.question(self, "Delete sample", "Remove this sample from SignBridge? Its webcam-captured file is deleted from SignBridge storage; imported original media is left untouched.") == QMessageBox.StandardButton.Yes:
            if sample: self._delete_captured_asset(sample["source_path"])
            self.db.delete_sample(sample_id); self.refresh_dataset()

    def start_camera(self):
        if self.camera and self.camera.isRunning(): return
        self.settings.set("camera_index", self.camera_index.value())
        self.model = None
        rec = self.db.latest_model()
        if rec and Path(rec["model_path"]).exists():
            try:
                model_path = Path(rec["model_path"]).resolve()
                if not model_path.is_relative_to((self.data_root / "models").resolve()): raise ValueError("Model is outside the local SignBridge model directory.")
                self.model = StaticClassifier.load(model_path, float(self.settings.get("confidence_threshold")), bool(self.settings.get("mirrored_input")), bool(self.settings.get("swap_handedness")))
            except Exception as exc: LOG.warning("Model load failed: %s", exc)
        self.camera = CameraWorker(self.camera_index.value(), int(self.settings.get("camera_width")), int(self.settings.get("camera_height")), self.landmark_toggle.isChecked(), self.model, float(self.settings.get("confidence_threshold")), int(self.settings.get("stability_frames")), int(self.settings.get("inference_fps")), self, mirrored_input=bool(self.settings.get("mirrored_input")), min_hand_confidence=float(self.settings.get("min_hand_confidence")), swap_handedness=bool(self.settings.get("swap_handedness")))
        self.camera.frame_ready.connect(self.show_frame); self.camera.prediction_ready.connect(self.show_prediction); self.camera.status.connect(self.camera_status.setText); self.camera.error.connect(self.show_camera_error)
        self.camera.hand_status.connect(self.hand_detection_status.setText)
        self.start_camera_btn.setEnabled(False); self.stop_camera_btn.setEnabled(True); self.camera.start()
        self.refresh_dashboard()

    def stop_camera(self):
        if self.camera:
            if self.camera.isRunning(): self.camera.stop()
            self.camera = None
        if hasattr(self, "start_camera_btn"): self.start_camera_btn.setEnabled(True); self.stop_camera_btn.setEnabled(False)
        if hasattr(self, "camera_status"): self.camera_status.setText("Camera stopped")
        if hasattr(self, "video_label"): self.video_label.setText("Camera is off\n\nSelect Start camera when ready"); self.video_label.setPixmap(QPixmap())
        self.refresh_dashboard()

    def show_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        if self.camera: self.camera.acknowledge_frame()

    def show_prediction(self, label, confidence, candidates, state, inference_ms, fps):
        self.current_prediction = label; self.current_confidence = confidence; self.current_candidates = candidates
        self.prediction_label.setText(label or "Uncertain / not recognized")
        self.confidence_label.setText(f"Model score {confidence:.0%} (uncalibrated)" if confidence else "Confidence —")
        self.recognition_status.setText(state)
        self.candidates_label.setText("Candidates: " + (" · ".join(f"{l} {s:.0%}" for l, s in candidates) if candidates else "none"))
        self.performance.setText(f"Approx. {fps:.1f} FPS  ·  inference {inference_ms:.1f} ms")

    def show_camera_error(self, text):
        self.camera_status.setText(text)
        LOG.warning("Camera: %s", text)

    def capture_sample(self):
        if not self.camera or not self.camera.isRunning(): QMessageBox.information(self, "Camera is off", "Start the camera first."); return
        if self.capture_class.currentData() is None: QMessageBox.information(self, "No class selected", "Create a gesture class in Dataset manager first."); return
        group = self.recording_group.text().strip()
        if not group: QMessageBox.warning(self, "Recording group required", "Set a recording group to support leakage-safe evaluation."); return
        frame = self.camera.latest_frame_copy()
        if frame is None: QMessageBox.warning(self, "No frame", "Wait for a camera frame and try again."); return
        try:
            import cv2
            label = self.capture_class.currentText().split("  ·")[0]
            folder = self.data_root / "dataset" / "raw" / f"class_{self.capture_class.currentData()}"
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f"capture_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            if not cv2.imwrite(str(target), frame): raise OSError("Could not save the camera image.")
            self.db.add_sample(label, target, "image", group, self.participant_group.text())
            self.refresh_dataset(); self.camera_status.setText(f"Saved one local sample for {label}.")
        except Exception as exc: QMessageBox.critical(self, "Sample capture failed", str(exc))

    def add_prediction(self):
        if not self.current_prediction:
            QMessageBox.information(self, "No stable prediction", "A sign must pass the confidence and temporal stability checks before it can be added."); return
        current = self.transcript_edit.toPlainText().strip()
        self.transcript_edit.setPlainText((current + " " + self.current_prediction).strip())

    def undo_word(self):
        words = self.transcript_edit.toPlainText().split()
        if words: self.transcript_edit.setPlainText(" ".join(words[:-1]))

    def clear_transcript(self): self.transcript_edit.clear()
    def copy_transcript(self): QApplication.clipboard().setText(self.transcript_edit.toPlainText())
    def normalize_transcript(self): self.transcript_edit.setPlainText(normalize_transcript(self.transcript_edit.toPlainText()))
    def save_transcript(self):
        text = self.transcript_edit.toPlainText().strip()
        if not text: QMessageBox.information(self, "Empty transcript", "Enter or confirm transcript text before saving."); return
        self.db.save_transcript(text); self.refresh_history(); QMessageBox.information(self, "Saved", "Transcript saved locally.")
    def speak_transcript(self):
        if not self.settings.get("speech_enabled"):
            QMessageBox.information(self, "Speech disabled", "Enable local speech in Settings & privacy."); return
        self.speech.rate = int(self.settings.get("speech_rate"))
        self.speech.speak(self.transcript_edit.toPlainText(), on_error=self.speech_error.emit)
    def _show_speech_error(self, text): QMessageBox.warning(self, "Local speech unavailable", text)

    def refresh_training_summary(self):
        if not hasattr(self, "training_dataset_summary"): return
        classes = self.db.list_classes(); samples = self.db.list_samples()
        by_class = ", ".join(f"{c['label']}: {c['sample_count']} samples" for c in classes) or "No classes yet"
        group_map = {}
        for sample in samples:
            group = sample["participant_group"].strip() or sample["recording_group"]
            group_map.setdefault(sample["label"], set()).add(group)
        group_counts = ", ".join(f"{cls['label']}: {len(group_map.get(cls['label'], set()))} groups" for cls in classes)
        under_sampled = [cls["label"] for cls in classes if len(group_map.get(cls["label"], set())) < 3]
        warning = ("\nInsufficient independent groups for: " + ", ".join(under_sampled) + ". Add samples across at least three separate recordings or participant groups.") if under_sampled else "\nGroup counts meet the minimum split pre-check."
        self.training_dataset_summary.setText(f"{len(classes)} classes · {len(samples)} media samples\n{by_class}\nIndependent groups: {group_counts}{warning}\nModel output: {self.data_root / 'models'}")

    def start_training(self):
        if self.training_worker and self.training_worker.isRunning(): return
        self.training_worker = TrainingWorker(self.db.path, self.data_root / "models", bool(self.settings.get("mirrored_input")), float(self.settings.get("min_hand_confidence")), bool(self.settings.get("swap_handedness")))
        self.training_worker.progress.connect(self.training_status.setText)
        self.training_worker.completed.connect(self.training_complete)
        self.training_worker.failed.connect(self.training_failed)
        self.train_btn.setEnabled(False); self.cancel_train_btn.setEnabled(True); self.training_progress.setVisible(True); self.training_status.setText("Starting training…")
        self.training_worker.start()

    def cancel_training(self):
        if self.training_worker: self.training_worker.cancel(); self.training_status.setText("Cancellation requested; waiting for the current file operation to finish…")

    def training_complete(self, result):
        self.train_btn.setEnabled(True); self.cancel_train_btn.setEnabled(False); self.training_progress.setVisible(False)
        self.model = StaticClassifier.load(result["model_path"], float(self.settings.get("confidence_threshold")), bool(self.settings.get("mirrored_input")), bool(self.settings.get("swap_handedness")))
        report = result["report"]
        val = report["validation"]; test = report["test"]
        self.training_status.setText(f"Training complete. Model saved locally: {result['model_path']}")
        self.training_metrics.setPlainText(json.dumps({"validation": {"sample_count": val["sample_count"], "accuracy": val["accuracy"], "balanced_accuracy": val["balanced_accuracy"]}, "held_out_test": {"sample_count": test["sample_count"], "accuracy": test["accuracy"], "balanced_accuracy": test["balanced_accuracy"], "classification_report": test["classification_report"], "confusion_matrix": test["confusion_matrix"], "class_order": test["class_order"]}, "usable_samples": report["usable_samples"], "failed_samples": report["failed_samples"], "confidence_note": report["confidence_note"]}, indent=2))
        self.refresh_dashboard()
        QMessageBox.information(self, "Training complete", f"Model trained and evaluated on independent local data. Held-out test accuracy: {test['accuracy']:.3f}. This is your dataset's result, not a general ISL accuracy claim.")

    def training_failed(self, error):
        self.train_btn.setEnabled(True); self.cancel_train_btn.setEnabled(False); self.training_progress.setVisible(False); self.training_status.setText("Training did not complete.")
        QMessageBox.warning(self, "Training unavailable", error)

    def refresh_history(self):
        if not hasattr(self, "history_table"): return
        entries = self.db.list_transcripts(); self.history_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            for col, value in enumerate((entry["created_at"], entry["transcript_text"])):
                item = QTableWidgetItem(str(value)); item.setData(Qt.ItemDataRole.UserRole, entry["id"]); self.history_table.setItem(row, col, item)

    def _selected_history(self):
        row = self.history_table.currentRow()
        return self.history_table.item(row, 0).data(Qt.ItemDataRole.UserRole) if row >= 0 else None

    def open_history(self):
        ident = self._selected_history()
        if ident is None: return
        entry = next((x for x in self.db.list_transcripts() if x["id"] == ident), None)
        if entry: self.navigate(1); self.transcript_edit.setPlainText(entry["transcript_text"])

    def export_history(self):
        ident = self._selected_history()
        if ident is None: return
        entry = next((x for x in self.db.list_transcripts() if x["id"] == ident), None)
        if not entry: return
        path, _ = QFileDialog.getSaveFileName(self, "Export transcript", "SignBridge-transcript.txt", "Text files (*.txt)")
        if path:
            Path(path).write_text(entry["transcript_text"], encoding="utf-8")

    def delete_history(self):
        ident = self._selected_history()
        if ident is not None: self.db.delete_transcript(ident); self.refresh_history()

    def clear_history(self):
        if QMessageBox.question(self, "Clear transcript history", "Delete all saved transcripts from this device?") == QMessageBox.StandardButton.Yes:
            self.db.clear_transcripts(); self.refresh_history()

    def save_settings(self):
        updates = {"camera_index": self.set_camera.value(), "camera_width": int(self.set_width.currentText()), "camera_height": int(self.set_height.currentText()), "inference_fps": self.set_inference_fps.value(), "min_hand_confidence": self.set_hand_confidence.value(), "mirrored_input": self.set_mirrored_input.isChecked(), "swap_handedness": self.set_swap_handedness.isChecked(), "confidence_threshold": self.set_threshold.value(), "stability_frames": self.set_stability.value(), "speech_enabled": self.set_speech.isChecked(), "speech_rate": self.set_rate.value(), "theme": self.set_theme.currentText(), "show_landmarks": self.landmark_toggle.isChecked() if hasattr(self, "landmark_toggle") else bool(self.settings.get("show_landmarks"))}
        for key, value in updates.items(): self.settings.set(key, value)
        self.camera_index.setValue(self.set_camera.value())
        self.apply_theme(self.settings.get("theme")); self._load_model(); QMessageBox.information(self, "Settings saved", "Settings are saved locally.")

    def reset_settings(self):
        if QMessageBox.question(self, "Reset settings", "Restore default application settings?") != QMessageBox.StandardButton.Yes: return
        self.settings.data = __import__("config.settings", fromlist=["defaults"]).defaults(); self.settings.save()
        self.set_camera.setValue(int(self.settings.get("camera_index"))); self.set_width.setCurrentText(str(self.settings.get("camera_width"))); self.set_height.setCurrentText(str(self.settings.get("camera_height")))
        self.set_inference_fps.setValue(int(self.settings.get("inference_fps"))); self.set_threshold.setValue(float(self.settings.get("confidence_threshold"))); self.set_stability.setValue(int(self.settings.get("stability_frames")))
        self.set_hand_confidence.setValue(float(self.settings.get("min_hand_confidence"))); self.set_mirrored_input.setChecked(bool(self.settings.get("mirrored_input")))
        self.set_swap_handedness.setChecked(bool(self.settings.get("swap_handedness")))
        self.set_speech.setChecked(bool(self.settings.get("speech_enabled"))); self.set_rate.setValue(int(self.settings.get("speech_rate"))); self.set_theme.setCurrentText(self.settings.get("theme")); self.landmark_toggle.setChecked(bool(self.settings.get("show_landmarks")))
        self.camera_index.setValue(int(self.settings.get("camera_index"))); self.apply_theme(self.settings.get("theme")); self._load_model()
        QMessageBox.information(self, "Settings reset", "Default preferences are restored. Existing samples and transcripts are unchanged.")

    def open_data_folder(self):
        try:
            if os.name == "nt": os.startfile(str(self.data_root))
            else: QMessageBox.information(self, "Local data folder", str(self.data_root))
        except Exception as exc: QMessageBox.warning(self, "Could not open folder", str(exc))

    def closeEvent(self, event):
        try:
            self.stop_camera(); self.speech.stop()
            if self.training_worker and self.training_worker.isRunning():
                self.training_worker.cancel(); self.training_worker.wait(4000)
        except Exception as exc: LOG.warning("Shutdown cleanup issue: %s", exc)
        event.accept()
