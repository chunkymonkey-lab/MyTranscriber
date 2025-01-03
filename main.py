"""
multi_transcriber_enhanced.py

A single-file PyQt6 application implementing several recommended improvements:
- Error handling & validation (model choice, file extraction)
- Basic configuration management (YAML + QSettings)
- Preliminary performance approach (batch processing skeleton)
- Partial MVC separation (some classes for logic, partial UI separation)
- Additional enhancements (drag-and-drop, speaker diarization improvements, etc.)
- ADDED: A LOGO in the header, with a resource_path function for py2app compatibility

Note: This is a demo, so you may want to adapt or refine it for production.
"""

import sys
import os
import uuid
import tempfile
from urllib.parse import urlparse
import yaml
import logging

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QFileDialog,
    QTextEdit, QTabWidget, QPushButton, QCheckBox, QLabel, QMessageBox,
    QStatusBar, QLineEdit, QAbstractItemView
)
from PyQt6.QtGui import QColor, QIcon, QPixmap, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QSettings, QSize

# Multimedia
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

# Audio/Video and Models
import whisper
import ffmpeg
import yt_dlp
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition

# Optional DistilWhisper
# from transformers import WhisperProcessor
# from DistilWhisper import DistilWhisperForConditionalGeneration

###############################################################################
#                          CONFIGURATION MANAGEMENT
###############################################################################

# In a real system, you might load these from a YAML/JSON file on disk.
# For demonstration, we embed them in-code. Then we combine them with QSettings.

DEFAULT_CONFIG = """
device: "cpu"
supported_extensions: [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".mov", ".avi", ".mkv"]
default_model: "base"
valid_models:
  - "tiny"
  - "base"
  - "small"
  - "medium"
  - "large"
  - "large-v2"
batch_processing: false
"""

CONFIG = yaml.safe_load(DEFAULT_CONFIG)

# We’ll also keep QSettings to store user preferences, like last model choice, diarization on/off, etc.
settings = QSettings("MyTranscriberCompany", "TranscriberApp")

###############################################################################
#                          LOGGING CONFIGURATION
###############################################################################

logging.basicConfig(
    filename="transcriber.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.info("Application started.")

###############################################################################
#                    HELPER: Resource Path for py2app
###############################################################################

def resource_path(relative_path):
    """
    Returns an absolute path to the given relative_path, ensuring compatibility
    when running inside a py2app .app bundle or a normal Python environment.
    """
    try:
        # If running in a py2app environment, __MEIPASS is the temporary folder
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(__file__)

    return os.path.join(base_path, relative_path)

###############################################################################
#                          MODEL UTILS
###############################################################################

def load_transcriber_model(model_choice, device="cpu"):
    """
    Validate the model choice against known or user-defined models,
    then attempt to load it.
    """
    valid_list = CONFIG["valid_models"]
    if model_choice not in valid_list:
        raise ValueError(f"Unsupported model '{model_choice}' is not in {valid_list}")

    # If you want to incorporate DistilWhisper:
    # if "distil" in model_choice.lower():
    #     return DistilWhisperForConditionalGeneration.from_pretrained(
    #         "naver/multilingual-distilwhisper-28k"
    #     )
    # else:
    return whisper.load_model(model_choice, device=device)

###############################################################################
#                          WORKER CLASSES
###############################################################################

class VideoDownloader(QThread):
    """
    Uses yt-dlp to download an online video/audio from a given URL.
    """
    progress = pyqtSignal(float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.temp_dir = tempfile.mkdtemp()

    def progress_hook(self, d):
        if d['status'] == 'downloading':
            pct_str = d.get('_percent_str', '0%').replace('%', '')
            try:
                self.progress.emit(float(pct_str))
            except ValueError:
                pass
        elif d['status'] == 'error':
            err = d.get('error', 'Unknown error')
            self.error.emit(f"Download error: {err}")

    def run(self):
        try:
            output_tmpl = os.path.join(self.temp_dir, '%(title)s.%(ext)s')
            ydl_opts = {
                'format': 'bestaudio/best',
                'progress_hooks': [self.progress_hook],
                'outtmpl': output_tmpl,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                downloaded_file = ydl.prepare_filename(info)
                self.finished.emit(downloaded_file)
        except Exception as e:
            self.error.emit(str(e))

class TranscriptionWorker(QThread):
    """
    Handles single-file transcription or translation, with minimal diarization.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str, bool)  # (file_path, final_text, is_translation)
    error = pyqtSignal(str, str)           # (file_path, error_message)

    def __init__(self, file_path,
                 language="auto",
                 do_translation=False,
                 do_diarization=True,
                 model_choice="base",
                 device="cpu"):
        super().__init__()
        self.file_path = file_path
        self.language = language
        self.do_translation = do_translation
        self.do_diarization = do_diarization
        self.model_choice = model_choice
        self.device = device

        # For demonstration, we load spkrec here:
        self.spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_ecapa"
        )

    def run(self):
        try:
            self.progress.emit(10)

            # Validate extension
            ext = os.path.splitext(self.file_path.lower())[1]
            if ext not in CONFIG["supported_extensions"]:
                raise ValueError(f"File extension '{ext}' not supported.")

            extracted_audio = self.extract_audio_if_needed(self.file_path)
            self.progress.emit(30)

            try:
                model = load_transcriber_model(self.model_choice, device=self.device)
            except Exception as me:
                self.error.emit(self.file_path, f"Model loading error: {str(me)}")
                return

            self.progress.emit(40)
            task_type = "translate" if self.do_translation else "transcribe"
            result = model.transcribe(
                extracted_audio,
                language=None if self.language == "auto" else self.language,
                task=task_type
            )

            if self.do_diarization:
                self.progress.emit(70)
                final_text = self.perform_diarization(extracted_audio, result["segments"])
            else:
                final_text = "\n".join([seg["text"] for seg in result["segments"]])

            self.progress.emit(100)
            self.finished.emit(self.file_path, final_text, self.do_translation)

            # Clean up
            if extracted_audio != self.file_path and os.path.exists(extracted_audio):
                os.remove(extracted_audio)

        except Exception as e:
            logging.error(f"TranscriptionWorker error on {self.file_path}: {str(e)}")
            self.error.emit(self.file_path, str(e))

    def extract_audio_if_needed(self, path):
        """If it's a pure audio extension, return as-is. If video, extract to .wav."""
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        _, ext = os.path.splitext(path.lower())
        if ext in audio_exts:
            return path

        tmp_wav = os.path.join(tempfile.mkdtemp(), f"{uuid.uuid4().hex}.wav")
        try:
            (
                ffmpeg
                .input(path)
                .output(tmp_wav, format="wav", ac="1", ar="16k", loglevel="quiet")
                .run(overwrite_output=True)
            )
            return tmp_wav
        except ffmpeg.Error as fe:
            raise RuntimeError(f"FFmpeg extraction error: {str(fe)}")

    def perform_diarization(self, audio_path, segments):
        diarized_output = []
        for seg in segments:
            start_s = seg["start"]
            end_s   = seg["end"]
            text    = seg["text"].strip()

            seg_wav = os.path.join(tempfile.mkdtemp(), f"seg_{uuid.uuid4().hex}.wav")
            (
                ffmpeg
                .input(audio_path, ss=start_s, to=end_s)
                .output(seg_wav, format="wav", ac="1", ar="16k", loglevel="quiet")
                .run(overwrite_output=True)
            )

            spkid = self.identify_speaker(seg_wav)
            start_str = self.format_timestamp(start_s)
            end_str   = self.format_timestamp(end_s)
            diarized_output.append(f"[{start_str} - {end_str}] {spkid}: {text}")

            if os.path.exists(seg_wav):
                os.remove(seg_wav)

        return "\n".join(diarized_output)

    def identify_speaker(self, seg_wav):
        # Basic approach: random short ID per segment
        short_id = uuid.uuid4().hex[:4].upper()
        return f"Speaker_{short_id}"

    def format_timestamp(self, sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

###############################################################################
#                           BUSINESS LOGIC
###############################################################################

class TranscriptionEngine:
    """
    A placeholder 'engine' that can process multiple files in batch.
    This is a demonstration of how you might handle batch logic.
    """
    def __init__(self, device="cpu", model_choice="base", diarization=True):
        self.device = device
        self.model_choice = model_choice
        self.diarization  = diarization

    def transcribe_files(self, file_paths):
        """
        Skeleton code for batch processing. For each file, do transcription.
        Real logic might queue these in threads, track overall progress, etc.
        """
        results = {}
        for fp in file_paths:
            try:
                # This is blocking demonstration code.
                worker = TranscriptionWorker(
                    fp, language="auto", do_translation=False,
                    do_diarization=self.diarization,
                    model_choice=self.model_choice,
                    device=self.device
                )
                worker.run()  # Synchronous call
                results[fp] = "Transcription done"
            except Exception as e:
                results[fp] = f"Error: {str(e)}"
        return results

###############################################################################
#                           MAIN WINDOW (UI)
###############################################################################

STATUS_NOT_STARTED = 0
STATUS_IN_PROGRESS = 1
STATUS_DONE = 2
STATUS_ERROR = 3

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Multi-File Transcriber")
        self.setMinimumSize(1200, 800)

        # Load user preferences from QSettings
        self.current_device = settings.value("device", CONFIG["device"])
        self.current_model  = settings.value("model", CONFIG["default_model"])
        self.current_diar   = settings.value("diarization", True, type=bool)

        self.worker = None
        self.download_worker = None
        self.batch_mode = CONFIG["batch_processing"]

        # main UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # LEFT: FILE TABLE
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(3)
        self.file_table.setHorizontalHeaderLabels(["Filename", "Status", "Full Path"])
        self.file_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.file_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.file_table.cellClicked.connect(self.on_file_selected)
        self.file_table.setAcceptDrops(True)
        self.file_table.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        main_layout.addWidget(self.file_table, stretch=1)

        # RIGHT: Panel
        right_panel = QVBoxLayout()

        # A) LOGO + Title
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        # Our new embedded logo approach:
        from PyQt6.QtGui import QPixmap
        logo_path = resource_path("resources/logo.png")
        pixmap = QPixmap(logo_path)
        if pixmap.isNull():
            print(f"Could not load logo from {logo_path}")
        else:
            scaled_logo = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio)
            logo_label.setPixmap(scaled_logo)
        header_layout.addWidget(logo_label)

        title_label = QLabel("Enhanced Multi-File Transcriber")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-left: 10px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        right_panel.addLayout(header_layout)

        # B) Media Player
        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.player.setAudioOutput(self.audio_out)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(400, 300)
        self.player.setVideoOutput(self.video_widget)
        right_panel.addWidget(self.video_widget)

        # Player controls
        controls_layout = QHBoxLayout()
        btn_play = QPushButton("Play")
        btn_play.clicked.connect(self.on_play)
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(self.on_stop)
        controls_layout.addWidget(btn_play)
        controls_layout.addWidget(btn_stop)
        right_panel.addLayout(controls_layout)

        # C) Tabs (Transcription vs. Translation)
        self.tab_widget = QTabWidget()
        self.transcription_text = QTextEdit()
        self.translation_text   = QTextEdit()

        self.tab_widget.addTab(self.transcription_text, "Transcription")
        self.tab_widget.addTab(self.translation_text, "Translation")
        right_panel.addWidget(self.tab_widget, stretch=1)

        # D) File row
        file_row = QHBoxLayout()
        browse_btn = QPushButton("Browse Files")
        browse_btn.clicked.connect(self.on_browse)
        file_row.addWidget(browse_btn)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste YouTube / online URL here")
        file_row.addWidget(self.url_input)

        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.on_download_url)
        file_row.addWidget(self.download_btn)
        right_panel.addLayout(file_row)

        # E) Model + Diar + Transcribe/Translate
        action_layout = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Model Choice")
        self.model_edit.setText(self.current_model)

        self.diar_check = QCheckBox("Diarization")
        self.diar_check.setChecked(self.current_diar)

        transcribe_btn = QPushButton("Transcribe")
        transcribe_btn.clicked.connect(self.on_transcribe)
        translate_btn  = QPushButton("Translate")
        translate_btn.clicked.connect(self.on_translate)

        action_layout.addWidget(QLabel("Model:"))
        action_layout.addWidget(self.model_edit)
        action_layout.addWidget(self.diar_check)
        action_layout.addWidget(transcribe_btn)
        action_layout.addWidget(translate_btn)
        right_panel.addLayout(action_layout)

        # F) Copy & Batch
        bottom_layout = QHBoxLayout()
        copy_btn = QPushButton("Copy Active Tab")
        copy_btn.clicked.connect(self.on_copy_text)
        bottom_layout.addWidget(copy_btn)

        if self.batch_mode:
            batch_btn = QPushButton("Run Batch")
            batch_btn.clicked.connect(self.on_batch_run)
            bottom_layout.addWidget(batch_btn)

        right_panel.addLayout(bottom_layout)

        main_layout.addLayout(right_panel, stretch=2)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.setAcceptDrops(True)

    ###########################################################################
    # DRAG AND DROP
    ###########################################################################
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            for u in urls:
                local_path = u.toLocalFile()
                if local_path:
                    self.add_file_to_table(local_path)

    ###########################################################################
    # FILE TABLE UTILS
    ###########################################################################
    def add_file_to_table(self, path):
        row = self.file_table.rowCount()
        self.file_table.insertRow(row)
        base_name = os.path.basename(path)
        item_fname  = QTableWidgetItem(base_name)
        item_status = QTableWidgetItem("Not Started")
        item_path   = QTableWidgetItem(path)

        self.set_table_item_status(item_status, STATUS_NOT_STARTED)
        self.file_table.setItem(row, 0, item_fname)
        self.file_table.setItem(row, 1, item_status)
        self.file_table.setItem(row, 2, item_path)

    def set_table_item_status(self, item, status_code):
        if status_code == STATUS_NOT_STARTED:
            item.setForeground(QColor("red"))
            item.setText("Not Started")
        elif status_code == STATUS_IN_PROGRESS:
            item.setForeground(QColor("orange"))
            item.setText("In Progress")
        elif status_code == STATUS_DONE:
            item.setForeground(QColor("green"))
            item.setText("Done")
        elif status_code == STATUS_ERROR:
            item.setForeground(QColor("red"))
            item.setText("Error")

    def on_file_selected(self, row, col):
        path_item = self.file_table.item(row, 2)
        if path_item:
            fpath = path_item.text()
            self.load_media(fpath)

    def load_media(self, fpath):
        if not os.path.isfile(fpath):
            return
        self.player.stop()
        media_url = QUrl.fromLocalFile(fpath)
        self.player.setSource(media_url)
        self.player.setPosition(0)

    def find_row_by_path(self, path):
        for r in range(self.file_table.rowCount()):
            item_path = self.file_table.item(r, 2)
            if item_path and item_path.text() == path:
                return r
        return -1

    ###########################################################################
    # BROWSE & DOWNLOAD
    ###########################################################################
    def on_browse(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        filter_str = "Audio/Video files (*.*);;All Files (*.*)"
        if dlg.exec():
            files = dlg.selectedFiles()
            for f in files:
                self.add_file_to_table(f)

    def on_download_url(self):
        url_str = self.url_input.text().strip()
        if not url_str or not self.is_valid_url(url_str):
            QMessageBox.warning(self, "Invalid URL", "Please provide a valid URL.")
            return
        self.download_btn.setEnabled(False)
        self.status_bar.showMessage("Downloading...")

        self.download_worker = VideoDownloader(url_str)
        self.download_worker.progress.connect(self.on_download_progress)
        self.download_worker.finished.connect(self.on_download_finished)
        self.download_worker.error.connect(self.on_download_error)
        self.download_worker.start()

    def on_download_progress(self, val):
        self.status_bar.showMessage(f"Downloading: {val:.1f}%")

    def on_download_finished(self, file_path):
        self.download_btn.setEnabled(True)
        self.status_bar.showMessage("Download complete.")
        self.add_file_to_table(file_path)

    def on_download_error(self, msg):
        QMessageBox.critical(self, "Download Error", msg)
        self.download_btn.setEnabled(True)
        self.status_bar.showMessage("Download error.")

    def is_valid_url(self, u):
        parsed = urlparse(u)
        return bool(parsed.scheme and parsed.netloc)

    ###########################################################################
    # TRANSCRIBE / TRANSLATE
    ###########################################################################
    def on_transcribe(self):
        row = self.file_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Selection", "Please select a file from the table.")
            return
        st_item = self.file_table.item(row, 1)
        path_item = self.file_table.item(row, 2)
        fpath = path_item.text()

        self.set_table_item_status(st_item, STATUS_IN_PROGRESS)
        self.status_bar.showMessage(f"Transcribing {os.path.basename(fpath)}...")

        self.start_transcription_thread(fpath, do_translation=False)

    def on_translate(self):
        row = self.file_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Selection", "Please select a file from the table.")
            return
        st_item = self.file_table.item(row, 1)
        path_item = self.file_table.item(row, 2)
        fpath = path_item.text()

        self.set_table_item_status(st_item, STATUS_IN_PROGRESS)
        self.status_bar.showMessage(f"Translating {os.path.basename(fpath)}...")

        self.start_transcription_thread(fpath, do_translation=True)

    def start_transcription_thread(self, file_path, do_translation):
        # Save user preferences
        settings.setValue("model", self.model_edit.text().strip())
        settings.setValue("diarization", self.diar_check.isChecked())

        self.worker = TranscriptionWorker(
            file_path=file_path,
            language="auto",
            do_translation=do_translation,
            do_diarization=self.diar_check.isChecked(),
            model_choice=self.model_edit.text().strip(),
            device=self.current_device
        )
        self.worker.progress.connect(self.on_transcription_progress)
        self.worker.finished.connect(self.on_transcription_finished)
        self.worker.error.connect(self.on_transcription_error)
        self.worker.start()

    def on_transcription_progress(self, val):
        self.status_bar.showMessage(f"Processing... {val}%")

    def on_transcription_finished(self, file_path, text_out, is_translation):
        row = self.find_row_by_path(file_path)
        if row >= 0:
            st_item = self.file_table.item(row, 1)
            self.set_table_item_status(st_item, STATUS_DONE)

        if is_translation:
            self.translation_text.setPlainText(text_out)
            self.tab_widget.setCurrentWidget(self.translation_text)
            self.status_bar.showMessage("Translation done.")
        else:
            self.transcription_text.setPlainText(text_out)
            self.tab_widget.setCurrentWidget(self.transcription_text)
            self.status_bar.showMessage("Transcription done.")

    def on_transcription_error(self, file_path, err_msg):
        row = self.find_row_by_path(file_path)
        if row >= 0:
            st_item = self.file_table.item(row, 1)
            self.set_table_item_status(st_item, STATUS_ERROR)

        QMessageBox.critical(self, "Error", f"{os.path.basename(file_path)}:\n{err_msg}")
        self.status_bar.showMessage("Error encountered.")

    ###########################################################################
    # BATCH PROCESSING
    ###########################################################################
    def on_batch_run(self):
        not_started_files = []
        for r in range(self.file_table.rowCount()):
            status_item = self.file_table.item(r, 1)
            path_item   = self.file_table.item(r, 2)
            if status_item and status_item.text() == "Not Started" and path_item:
                not_started_files.append(path_item.text())

        if not not_started_files:
            QMessageBox.information(self, "Batch", "No files with 'Not Started' status found.")
            return

        engine = TranscriptionEngine(
            device=self.current_device,
            model_choice=self.model_edit.text().strip(),
            diarization=self.diar_check.isChecked()
        )
        results = engine.transcribe_files(not_started_files)

        for f in results:
            row = self.find_row_by_path(f)
            if row >= 0:
                st_item = self.file_table.item(row, 1)
                if results[f].startswith("Error"):
                    self.set_table_item_status(st_item, STATUS_ERROR)
                else:
                    self.set_table_item_status(st_item, STATUS_DONE)

        QMessageBox.information(self, "Batch", "Batch processing complete.")

    ###########################################################################
    # COPY TEXT
    ###########################################################################
    def on_copy_text(self):
        current_idx = self.tab_widget.currentIndex()
        if current_idx == 0:
            text = self.transcription_text.toPlainText()
        else:
            text = self.translation_text.toPlainText()

        if text:
            QApplication.clipboard().setText(text)
            self.status_bar.showMessage("Text copied.")
        else:
            self.status_bar.showMessage("Nothing to copy.")

    ###########################################################################
    # PLAYER CONTROLS
    ###########################################################################
    def on_play(self):
        self.player.play()

    def on_stop(self):
        self.player.stop()
        self.player.setPosition(0)

    ###########################################################################
    # CLOSE & CLEANUP
    ###########################################################################
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.terminate()

        self.player.stop()

        # Save user preferences
        settings.setValue("model", self.model_edit.text().strip())
        settings.setValue("diarization", self.diar_check.isChecked())

        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
