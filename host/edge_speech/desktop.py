"""Windows desktop recording and transcription client for the INT8 layer service."""
from datetime import datetime
from pathlib import Path
import audioop
import sys
from threading import Thread
from uuid import uuid4

from PySide6.QtCore import Qt, QSettings, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from .layer_stream import BOARD_HOST, BOARD_PIN, connect, transcribe, transcribe_segments
from .local_capture import Pcm16, WavSpool, adaptive_segments, pause_segments, wav_blocks
from .local_qwen import Qwen

PCM_KINDS = {
    QAudioFormat.Int16: "int16", QAudioFormat.Int32: "int32",
    QAudioFormat.Float: "float", QAudioFormat.UInt8: "uint8",
}

STYLE = """
QWidget { background:#f4f5f2; color:#172522; font-family:"Microsoft YaHei UI"; font-size:13px; }
QLabel#title { font-size:26px; font-weight:700; }
QLabel#status { padding:9px 12px; background:#e6eeea; color:#225448; border-radius:6px; }
QLabel#status[error="true"] { background:#fbece8; color:#842e20; }
QLineEdit, QComboBox, QTextEdit { border:1px solid #bfc9c3; border-radius:6px; background:white; padding:7px 9px; }
QTextEdit { padding:18px; font-size:17px; }
QPushButton { background:#ffffff; border:1px solid #bcc9bf; border-radius:6px; padding:9px 14px; }
QPushButton:hover { background:#e8f0ec; }
QPushButton#primary { background:#135e50; color:white; border:1px solid #135e50; font-weight:600; }
QPushButton#primary:hover { background:#104b40; }
QPushButton:disabled, QPushButton#primary:disabled { background:#e3e7e3; color:#66736e; border-color:#ccd4ce; }
QPushButton:focus, QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border:1px solid #11816b; }
QProgressBar { background:#e0e6e0; border-radius:3px; min-height:6px; max-height:6px; }
QProgressBar::chunk { background:#288c75; border-radius:3px; }
"""


class Desktop(QWidget):
    ready = Signal()
    partial = Signal(str)
    completed = Signal(str, int)
    failed = Signal(str)
    polished = Signal(str, object)
    polish_error = Signal(str)

    def __init__(self, archive=None, connector=connect, qwen_folder=None):
        super().__init__()
        self.settings = QSettings("Edge Speech", "FZ3B Transcription")
        self.archive = archive or Path.home() / "Documents" / "FZ3B 转写"
        self.connector = connector
        self.busy = False
        bundle = qwen_folder or Path(__file__).resolve().parents[2]/"qwen"
        self.qwen = Qwen(bundle) if (bundle/"manifest.json").is_file() else None
        self.polishing = False
        self.final_text = self.polished_text = ""
        self.spool = None
        self.source = self.device = self.audio_path = None
        self.audio_format = None
        self.setWindowTitle("FZ3B · 本地语音转写")
        self.setMinimumSize(650, 540)
        self.resize(820, 630)
        self.setStyleSheet(STYLE)
        self.build()
        self.media = QMediaDevices(self)
        self.media.audioInputsChanged.connect(self.refresh_mics)
        self.refresh_mics()
        self.ready.connect(self.connected)
        self.partial.connect(self.show_partial)
        self.completed.connect(self.finish_result)
        self.failed.connect(self.show_failure)
        self.polished.connect(self.save_polished)
        self.polish_error.connect(self.show_polish_error)

    def build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(27, 23, 27, 24)
        layout.setSpacing(12)
        title = QLabel("语音转写")
        title.setObjectName("title")
        layout.addWidget(title)
        self.status = QLabel("就绪 · 选择麦克风或打开录音")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.mics = QComboBox()
        self.host = QLineEdit(self.settings.value("board", BOARD_HOST))
        self.user = QLineEdit(self.settings.value("user", "root"))
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText("留空使用 SSH 代理")
        connection = QHBoxLayout()
        for w, stretch in ((self.host, 2), (self.user, 1), (self.password, 2)):
            connection.addWidget(w, stretch)
        self.host.setAccessibleName("板卡地址")
        self.user.setAccessibleName("SSH 用户名")
        self.password.setAccessibleName("SSH 密码")
        form = QFormLayout()
        form.addRow("板卡地址", connection)
        form.addRow("USB 麦克风", self.mics)
        self.segmentation = QComboBox()
        self.segmentation.addItem("智能分段 · 中英混合", "adaptive")
        self.segmentation.addItem("连续 · 保留上下文", "continuous")
        self.segmentation.addItem("长停顿后分段", "legacy")
        selected = self.segmentation.findData(self.settings.value("segmentation", "adaptive"))
        self.segmentation.setCurrentIndex(max(0, selected))
        self.segmentation.setToolTip("智能分段随环境底噪调整，并在停顿后重新起句。录音始终完整保存。")
        form.addRow("转写模式", self.segmentation)
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.record = QPushButton("开始录音")
        self.record.setObjectName("primary")
        self.record.clicked.connect(self.record_clicked)
        self.open_wav = QPushButton("打开 WAV")
        self.open_wav.clicked.connect(self.pick_wav)
        self.elapsed = QLabel("00:00")
        self.elapsed.setMinimumWidth(48)
        actions.addWidget(self.record)
        actions.addWidget(self.open_wav)
        actions.addStretch(1)
        actions.addWidget(self.elapsed, alignment=Qt.AlignRight)
        layout.addLayout(actions)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setAccessibleName("麦克风音量")
        layout.addWidget(self.meter)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setAccessibleName("转写原稿")
        self.text.setPlaceholderText("录音开始后，板卡返回的原稿会显示在这里。")
        layout.addWidget(self.text, 1)
        bottom = QHBoxLayout()
        self.copy = QPushButton("复制原稿")
        self.copy.setEnabled(False)
        self.copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        self.listen = QPushButton("打开录音")
        self.listen.setEnabled(False)
        self.listen.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.audio_path))))
        folder = QPushButton("打开保存位置")
        folder.clicked.connect(self.open_archive)
        for w in (self.copy, self.listen, folder):
            bottom.addWidget(w)
        bottom.addStretch(1)
        layout.addLayout(bottom)
        if self.qwen:
            editing = QHBoxLayout()
            self.auto_polish = QCheckBox("转写后用本地 Qwen 整理")
            self.auto_polish.setChecked(True)
            self.polish_button = QPushButton("整理原稿")
            self.polish_button.setEnabled(False)
            self.polish_button.clicked.connect(self.request_polish)
            self.compare = QPushButton("查看整理稿")
            self.compare.setEnabled(False)
            self.compare.clicked.connect(self.view_polished)
            for widget in (self.auto_polish, self.polish_button, self.compare):
                editing.addWidget(widget)
            editing.addStretch(1)
            layout.addLayout(editing)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=lambda: self.record.click())
        QShortcut(QKeySequence.StandardKey.Open, self, activated=lambda: self.open_wav.click())

    def set_status(self, text, error=False):
        self.status.setText(text)
        self.status.setProperty("error", error)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def refresh_mics(self):
        current = self.mics.currentData()
        known = QMediaDevices.audioInputs()
        if self.spool and current and not any(x.id() == current.id() for x in known):
            self.stop_capture()
            self.set_status("麦克风已断开 · 正在整理已录音频", True)
        self.mics.clear()
        preferred = 0
        for index, device in enumerate(known):
            self.mics.addItem(device.description(), device)
            if (current and current.id() == device.id()) or (not current and "UGREEN Camera" in device.description()):
                preferred = index
        self.mics.setCurrentIndex(preferred)
        self.controls()

    def controls(self):
        self.record.setEnabled(bool(self.spool or (not self.busy and not self.polishing and self.mics.count())))
        self.open_wav.setEnabled(not self.busy and not self.polishing and self.spool is None)
        for w in (self.host, self.user, self.password, self.mics, self.segmentation):
            w.setEnabled(not self.busy and self.spool is None)
        if self.qwen:
            self.polish_button.setEnabled(self.polishing or (not self.busy and bool(self.final_text)))
            self.compare.setEnabled(bool(self.polished_text))

    def archive_paths(self):
        self.archive.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-" + uuid4().hex[:6]
        return self.archive / (stamp + ".wav"), self.archive / (stamp + ".txt")

    def record_clicked(self):
        if self.spool:
            self.stop_capture()
            return
        try:
            self.start_capture()
        except Exception as exc:
            if self.spool:
                self.stop_capture()
            self.show_failure(str(exc))

    def start_capture(self):
        selected = self.mics.currentData()
        if self.busy or selected is None:
            return
        self.audio_path, self.text_path = self.archive_paths()
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Int16)
        if not selected.isFormatSupported(fmt):
            fmt = selected.preferredFormat()
        self.audio_format = Pcm16(fmt.sampleRate(), fmt.channelCount(), PCM_KINDS[fmt.sampleFormat()])
        self.spool = WavSpool(self.audio_path)
        self.source = QAudioSource(selected, fmt, self)
        self.source.setBufferSize(fmt.bytesForDuration(200_000))
        self.device = self.source.start()
        if not self.device or self.source.error() != QAudio.Error.NoError:
            raise RuntimeError("无法打开所选麦克风，请检查 Windows 麦克风权限")
        self.device.readyRead.connect(self.capture)
        self.source.stateChanged.connect(self.capture_state)
        self.listen.setEnabled(False)
        self.record.setText("停止录音")
        self.elapsed.setText("00:00")
        self.start_worker(self.spool.blocks())

    def capture(self):
        if self.device is None or self.spool is None:
            return
        try:
            data = self.audio_format.push(bytes(self.device.readAll()))
            self.spool.push(data)
            seconds = self.spool.size // 32000
            self.elapsed.setText(f"{seconds//60:02d}:{seconds%60:02d}")
            if data:
                self.meter.setValue(min(100, round(audioop.rms(data, 2)**0.5)))
        except (OSError, ValueError) as exc:
            self.stop_capture(drain=False)
            self.set_status(f"录音写入失败 · {exc}", True)

    def capture_state(self, state):
        if self.source and state == QAudio.State.StoppedState and self.source.error() != QAudio.Error.NoError:
            self.stop_capture()
            self.set_status("麦克风停止 · 已录制的音频已保存", True)

    def stop_capture(self, drain=True):
        if drain and self.device:
            self.capture()
        source = self.source
        spool = self.spool
        self.source = self.device = self.spool = None
        if source:
            source.stop()
            source.deleteLater()
        if spool:
            try:
                spool.finish()
            except OSError as exc:
                self.set_status(f"录音文件写入失败 · {exc}", True)
            self.listen.setEnabled(spool.size > 0)
        self.meter.setValue(0)
        self.record.setText("开始录音")
        self.controls()
        if self.busy:
            self.set_status("录音已停止 · 正在接收最终原稿")

    def pick_wav(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择录音", "", "WAV 音频 (*.wav)")
        if path:
            self.use_wav(Path(path))

    def use_wav(self, path):
        self.audio_path = path
        _, self.text_path = self.archive_paths()
        self.listen.setEnabled(True)
        self.start_worker(wav_blocks(path))

    def start_worker(self, blocks):
        self.busy = True
        self.final_text = self.polished_text = ""
        self.text.clear()
        self.copy.setEnabled(False)
        self.controls()
        self.set_status("正在连接 FZ3B" + (" · 本机正在录音" if self.spool else ""))
        host, user, password = self.host.text().strip(), self.user.text().strip(), self.password.text()
        mode = self.segmentation.currentData()
        self.settings.setValue("board", host)
        self.settings.setValue("user", user)
        self.settings.setValue("segmentation", mode)
        def run():
            try:
                with self.connector(host, user, password, BOARD_PIN) as ssh:
                    if mode == "continuous":
                        text, samples = transcribe(ssh, blocks, self.partial.emit, self.ready.emit)
                    else:
                        segment = adaptive_segments if mode == "adaptive" else pause_segments
                        text, samples = transcribe_segments(ssh, segment(blocks), self.partial.emit, self.ready.emit)
                self.completed.emit(text, samples)
            except Exception as exc:
                self.failed.emit(str(exc))
        Thread(target=run, daemon=True).start()

    def connected(self):
        self.set_status("正在录音和转写" if self.spool else "板卡正在转写")

    def show_partial(self, text):
        self.text.setPlainText(text)
        bar = self.text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def finish_result(self, text, samples):
        self.busy = False
        self.final_text = text
        self.controls()
        self.show_partial(text)
        self.copy.setEnabled(bool(text))
        try:
            self.text_path.write_text(text + "\n" if text else "", encoding="utf-8")
            self.set_status(f"转写完成 · 板卡已接收 {samples/16000:.2f} 秒音频" if text
                            else f"未检测到语音 · 板卡已接收 {samples/16000:.2f} 秒音频")
            if text and self.qwen and self.auto_polish.isChecked():
                self.request_polish()
        except OSError as exc:
            self.set_status(f"原稿仍可复制 · 写入文件失败 · {exc}", True)

    def show_failure(self, message):
        self.busy = False
        self.controls()
        self.set_status(f"转写连接失败 · {message}" + (" · 本机继续录音" if self.spool else ""), True)

    def request_polish(self):
        if self.polishing:
            self.qwen.close()
            self.set_status("正在取消整理 · 原稿可复制")
            return
        text = self.final_text
        if not text or self.busy:
            return
        self.qwen.cancel.clear()
        self.polishing = True
        self.polish_button.setText("取消整理")
        self.set_status("原稿已保存 · 本地 Qwen 正在整理")
        self.controls()

        def run():
            try:
                proposal, info = self.qwen.polish(text)
                if self.qwen.cancel.is_set():
                    self.polish_error.emit("已取消整理")
                else:
                    self.polished.emit(proposal, info)
            except Exception as exc:
                self.polish_error.emit("已取消整理" if self.qwen.cancel.is_set() else str(exc))
        Thread(target=run, daemon=True).start()

    def polishing_finished(self):
        self.polishing = False
        self.polish_button.setText("整理原稿")
        self.controls()

    def save_polished(self, text, info):
        if self.qwen.cancel.is_set():
            self.show_polish_error("已取消整理")
            return
        self.polishing_finished()
        if text == self.final_text:
            self.set_status(f"整理完成 · 原稿保留（{info['total_s']:.2f} 秒）")
            return
        path = self.text_path.with_suffix(".整理.txt")
        self.polished_text = text
        self.compare.setEnabled(True)
        try:
            path.write_text(text+"\n", encoding="utf-8")
            self.set_status(f"整理稿已另存 · 可查看与原稿的对照（{info['total_s']:.2f} 秒）")
        except OSError as exc:
            self.set_status(f"整理稿仍可查看 · 写入文件失败 · {exc}", True)

    def show_polish_error(self, message):
        self.polishing_finished()
        self.set_status(f"{message} · 原稿已保留", True)

    def view_polished(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("原稿与整理稿")
        dialog.resize(760, 420)
        layout = QVBoxLayout(dialog)
        views = QHBoxLayout()
        for name, text in (("原稿", self.final_text), ("整理稿", self.polished_text)):
            column = QVBoxLayout()
            column.addWidget(QLabel(name))
            field = QTextEdit()
            field.setReadOnly(True)
            field.setPlainText(text)
            column.addWidget(field)
            views.addLayout(column)
        layout.addLayout(views)
        copy = QPushButton("复制整理稿")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.polished_text))
        layout.addWidget(copy, alignment=Qt.AlignRight)
        dialog.exec()

    def open_archive(self):
        try:
            self.archive.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.archive)))
        except OSError as exc:
            self.set_status(f"无法访问保存位置 · {exc}", True)

    def closeEvent(self, event):
        if self.spool:
            self.stop_capture()
        if self.busy:
            self.set_status("正在收尾 · 收到最终原稿后可以关闭窗口")
            event.ignore()
        else:
            if self.qwen:
                self.qwen.close()
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FZ3B 转写")
    window = Desktop()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
