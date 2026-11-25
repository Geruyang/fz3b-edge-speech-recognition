"""Windows Qt session view for the board's FZAS preview and transcript."""

from __future__ import annotations

import argparse
from collections import deque
import logging
from pathlib import Path
from queue import Empty, Queue
import struct
from threading import Event, Thread
from time import perf_counter_ns
from uuid import uuid4

from PySide6.QtCore import QBuffer, QIODevice, QObject, QPointF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .client import FzasClient, HostKeyMismatch, ProtocolError, decode_audio, frame_json
from .types import AudioBlock, MessageType, TranscriptEvent, validate_transcript_event


LOG = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
WAVEFORM_SAMPLES = 10 * SAMPLE_RATE
PREVIEW_SAMPLES = 120 * SAMPLE_RATE
STATES = {
    "ready": "就绪", "recording": "录音中", "draining": "正在完成转写",
    "finalized": "原稿已冻结", "refining": "整理中", "completed": "已完成",
    "faulted": "板端故障",
}
ERRORS = {
    "busy": "板端正忙", "invalid_state": "当前状态无法执行此操作", "model_mismatch": "模型版本不匹配",
    "capture_unavailable": "板端麦克风不可用", "storage_low": "板端存储空间不足",
    "accelerator_unavailable": "板端加速器不可用", "drain_timeout": "排空超时",
    "storage_error": "录音保存失败，请检查板端存储空间",
    "capture_storage_error": "录音保存失败，请检查板端存储空间并保留已保存的音频",
    "capture_error": "采集或读取录音失败", "drain_failed": "剩余录音转写失败",
}
BACKENDS = {"cpu": "CPU", "pl": "FPGA（PL）", "mixed": "CPU + FPGA"}
PCM_FORMAT = {"sample_rate_hz": SAMPLE_RATE, "channels": 1, "encoding": "s16le"}
CAPTURE_GAP_REASONS = {"xrun": "设备缓冲溢出", "device_xrun": "设备缓冲溢出",
                       "device_disconnect": "设备断开", "disconnect": "设备断开", "timeout": "设备超时"}


def _time_at(sample: int) -> str:
    milliseconds = sample * 1000 // SAMPLE_RATE
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class _UnsupportedAudioFormat(ValueError):
    pass


class _Transport(QObject):
    connected = Signal(object)
    received = Signal(object, object, object)
    failed = Signal(object, str)

    def __init__(self, parent, connect_client):
        super().__init__(parent)
        self.connect_client = connect_client
        self.stopping = Event()
        self.requests = Queue()

    def open(self):
        Thread(target=self._run, name="fzas-requests", daemon=True).start()

    def close(self):
        self.stopping.set()

    def _run(self):
        client = None
        try:
            client = self.connect_client()
            readers = (client.receive_control, client.receive_archive)
            for channel, read in zip(("control", "archive"), readers, strict=True):
                if self.stopping.is_set():
                    return
                hello = read()
                if hello.message_type != MessageType.HELLO:
                    raise ProtocolError("FZAS channel did not start with HELLO")
                data = frame_json(hello)
                if data.get("protocol") != 1 or data.get("channel") != channel:
                    raise ProtocolError("unexpected FZAS HELLO version or channel")
                if data.get("audio_format") != PCM_FORMAT:
                    raise _UnsupportedAudioFormat
            if self.stopping.is_set():
                return
            self.connected.emit(self)
            for reader in readers:
                Thread(target=self._receive, args=(reader,), daemon=True).start()
            while not self.stopping.is_set():
                try:
                    operation, request_id, fields = self.requests.get(timeout=0.1)
                except Empty:
                    continue
                client.send_request(operation, request_id=request_id, **fields)
        except Exception as exc:
            self._report_failure(exc)
        finally:
            if client is not None:
                client.close()

    def _receive(self, read):
        try:
            while not self.stopping.is_set():
                try:
                    frame = read()
                except TimeoutError:
                    continue
                received_ns = perf_counter_ns()
                if frame.message_type == MessageType.PCM:
                    payload = decode_audio(frame)
                elif frame.message_type in (MessageType.RESPONSE, MessageType.EVENT):
                    payload = frame_json(frame)
                else:
                    continue
                if not self.stopping.is_set():
                    self.received.emit(self, (frame.message_type, payload), received_ns)
        except Exception as exc:
            self._report_failure(exc)

    def _report_failure(self, exc):
        if not self.stopping.is_set():
            self.stopping.set()
            LOG.warning("FZAS connection ended (%s)", type(exc).__name__)
            if isinstance(exc, HostKeyMismatch):
                message = "SSH 主机指纹不匹配，请核对板端身份。"
            elif isinstance(exc, _UnsupportedAudioFormat):
                message = "板端音频格式不匹配，需使用 16 kHz 单声道 S16_LE PCM。"
            elif isinstance(exc, ProtocolError):
                message = "FZAS 通信协议错误，请检查板端服务版本后重新连接。"
            else:
                message = "板端连接中断，请检查地址、SSH 身份及板端服务后重新连接。"
            self.failed.emit(self, message)


class _Waveform(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(88)
        self.setAccessibleName("板端单声道音频波形，最近十秒")
        self.traces = []
        self.gaps = []
        self.sample_end = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#122334"))
        width, height = self.width(), self.height()
        sample_start = max(0, self.sample_end - WAVEFORM_SAMPLES)

        def x_at(sample):
            return width * (sample - sample_start) / WAVEFORM_SAMPLES

        for start, end in self.gaps:
            painter.fillRect(round(x_at(max(sample_start, start))), 0,
                             max(2, round(x_at(end) - x_at(max(sample_start, start)))), height,
                             QColor("#704035"))
        painter.setPen(QPen(QColor("#2b4052")))
        for grid in range(11):
            x = width * grid / 10
            painter.drawLine(QPointF(x, 0), QPointF(x, height))
        painter.drawLine(QPointF(0, height / 2), QPointF(width, height / 2))
        painter.setPen(QPen(QColor("#73e4ba"), 1.3))
        for start, end, lower, upper in self.traces:
            x = x_at((start + end) / 2)
            radius = height * 0.46 / 32768
            painter.drawLine(QPointF(x, height / 2 - upper * radius),
                             QPointF(x, height / 2 - lower * radius + 1))


class _TranscriptTable(QTableWidget):
    painted = Signal()

    def paintEvent(self, event):
        super().paintEvent(event)
        self.painted.emit()


class SessionView(QWidget):
    """Show one active board session; apply/audio callbacks may run on a worker."""

    _event_ready = Signal(object, object)
    _audio_ready = Signal(object)

    def __init__(self, *, connect_client=None, start_options=None, connection_name="FZAS"):
        super().__init__()
        self.connect_client = connect_client
        self.connection_name = connection_name
        self.start_options = dict(start_options or {})
        self.session_id = None
        self.pending_request_id = None
        self._pending_operation = None
        self._pending_fields = None
        self.first_text_received_ns = None
        self.first_text_visible_ns = None
        self._transport = None
        self._state = "ready"
        self._connected = False
        self._job_id = None
        self._event_watermark = None
        self._segments = {}
        self._row_by_segment = {}
        self._rows = []
        self._audio = deque()
        self._early_audio = deque(maxlen=100)
        self._peaks = deque()
        self._gaps = deque()
        self._sample_end = 0
        self._gap_count = 0
        self._capture_gap_count = 0
        self._preview_dropped_samples = 0
        self._stop_sample = None
        self._processed_sample = None
        self._async_stop_supported = False
        self._capture_storage_failed = False
        self._wave_dirty = True
        self._buffer = None
        self._build_widgets()
        self._audio_output = None
        self._event_ready.connect(self._apply)
        self._audio_ready.connect(self._set_audio)
        self._set_state("ready")

    def _build_widgets(self):
        self.setWindowTitle("FZ3B 语音转写")
        self.resize(980, 720)
        self.setMinimumSize(720, 600)
        self.setFont(QFont("Microsoft YaHei UI", 10))
        self.setObjectName("sessionView")
        self.setStyleSheet("""
            QWidget#sessionView { background: #edf2f5; color: #102636; }
            QLabel { color: #183345; }
            QLabel#title { font-size: 26px; font-weight: 700; }
            QLabel#section { font-size: 15px; font-weight: 700; }
            QLabel#error { color: #952a26; }
            QLabel#status { color: #214b43; }
            QFrame#panel { background: #ffffff; border: 1px solid #c9d6df; border-radius: 7px; }
            QTableWidget, QPlainTextEdit { background: #ffffff; color: #142c3d; border: 1px solid #c9d6df;
                selection-background-color: #d8eee6; selection-color: #142c3d; }
            QHeaderView::section { background: #edf2f5; color: #284555; border: 0; padding: 6px; }
            QPushButton { background: #ffffff; border: 1px solid #a8b9c5; border-radius: 5px;
                color: #153342; padding: 5px 12px; min-height: 25px; }
            QPushButton:hover { background: #e3eee9; }
            QPushButton#primary { background: #006454; border-color: #006454; color: white; }
            QPushButton#primary:hover { background: #007664; }
            QPushButton:disabled, QPushButton#primary:disabled {
                background: #e7ebef; border-color: #cfdae2; color: #556778; }
            QPushButton:focus, QPushButton#primary:focus { border: 2px solid #377fc9; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(11)

        header = QHBoxLayout()
        title = QLabel("FZ3B 语音转写")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        self.connection_status = QLabel("未连接")
        self.connection_status.setObjectName("status")
        header.addWidget(self.connection_status)
        layout.addLayout(header)

        controls = QHBoxLayout()
        self.connect_button = QPushButton("连接板卡")
        self.start_button = QPushButton("开始录音")
        self.start_button.setObjectName("primary")
        self.stop_button = QPushButton("结束录音")
        self.cancel_button = QPushButton("取消整理")
        self.session_status = QLabel()
        self.session_status.setObjectName("status")
        for button in (self.connect_button, self.start_button, self.stop_button, self.cancel_button):
            controls.addWidget(button)
        controls.addStretch()
        controls.addWidget(self.session_status)
        layout.addLayout(controls)
        self.error_status = QLabel()
        self.error_status.setObjectName("error")
        self.error_status.setWordWrap(True)
        self.error_status.hide()
        layout.addWidget(self.error_status)

        wave_panel = QFrame()
        wave_panel.setObjectName("panel")
        wave_layout = QVBoxLayout(wave_panel)
        wave_layout.setContentsMargins(12, 9, 12, 9)
        wave_head = QHBoxLayout()
        wave_title = QLabel("板端音频波形")
        wave_title.setObjectName("section")
        wave_head.addWidget(wave_title)
        wave_head.addStretch()
        wave_head.addWidget(QLabel("16 kHz  ·  单声道  ·  10 秒视窗"))
        wave_layout.addLayout(wave_head)
        self.waveform = _Waveform()
        wave_layout.addWidget(self.waveform)
        self.waveform_status = QLabel()
        self.waveform_status.setWordWrap(True)
        wave_layout.addWidget(self.waveform_status)
        self.capture_gap_status = QLabel()
        self.capture_gap_status.setObjectName("error")
        self.capture_gap_status.setTextFormat(Qt.TextFormat.PlainText)
        self.capture_gap_status.setWordWrap(True)
        self.capture_gap_status.hide()
        wave_layout.addWidget(self.capture_gap_status)
        layout.addWidget(wave_panel)

        transcript_head = QHBoxLayout()
        transcript_title = QLabel("识别原稿")
        transcript_title.setObjectName("section")
        transcript_head.addWidget(transcript_title)
        transcript_head.addStretch()
        self.backend_status = QLabel("推理模式待确认")
        self.backend_status.setTextFormat(Qt.TextFormat.PlainText)
        transcript_head.addWidget(self.backend_status)
        layout.addLayout(transcript_head)

        self.transcript_table = _TranscriptTable(0, 3)
        self.transcript_table.setAccessibleName("识别原稿片段，按样本起点试听")
        self.transcript_table.setHorizontalHeaderLabels(("片段起点", "状态", "原文"))
        self.transcript_table.verticalHeader().hide()
        self.transcript_table.verticalHeader().setDefaultSectionSize(30)
        self.transcript_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.transcript_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.transcript_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.transcript_table.setWordWrap(False)
        columns = self.transcript_table.horizontalHeader()
        columns.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        columns.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        columns.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        columns.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.transcript_table.painted.connect(self._record_first_visible)
        layout.addWidget(self.transcript_table, 1)

        self.segment_text = QPlainTextEdit()
        self.segment_text.setAccessibleName("所选片段的完整原文")
        self.segment_text.setReadOnly(True)
        self.segment_text.setPlaceholderText("选择片段查看完整文字")
        self.segment_text.setFixedHeight(79)
        layout.addWidget(self.segment_text)
        playback = QHBoxLayout()
        self.play_button = QPushButton("试听所选片段")
        self.playback_status = QLabel("可试听最近 120 秒内已收到的板端音频")
        self.playback_status.setWordWrap(True)
        playback.addWidget(self.play_button)
        playback.addWidget(self.playback_status, 1)
        layout.addLayout(playback)

        self.connect_button.clicked.connect(self._connect)
        self.start_button.clicked.connect(self._start_or_complete)
        self.stop_button.clicked.connect(self._stop_recording)
        self.cancel_button.clicked.connect(
            lambda: self._request("cancel_refine", session_id=self.session_id, refinement_job_id=self._job_id))
        self.play_button.clicked.connect(self._play_selected)
        self.transcript_table.currentCellChanged.connect(self._show_selected)
        self._show_selected()
        self.waveform_refresh = QTimer(self)
        self.waveform_refresh.timeout.connect(self.refresh_waveform)
        self.waveform_refresh.start(66)
        self.refresh_waveform()

    def _set_state(self, state):
        self._state = state
        status = STATES.get(state, "状态待确认")
        if state == "draining" and self._capture_storage_failed:
            status = "录音保存失败，正在收尾"
            if self._stop_sample is not None:
                status += f"  ·  录音止于 {_time_at(self._stop_sample)}"
        elif state == "recording" and self.pending_request_id and self._pending_operation == "stop":
            status = "正在停止录音…"
        elif state == "draining" and self._processed_sample is not None and self._stop_sample is not None:
            status += f"  ·  已处理 {_time_at(self._processed_sample)} / {_time_at(self._stop_sample)}"
        elif self._stop_sample is not None and state not in ("ready", "recording"):
            status += f"  ·  停采 {_time_at(self._stop_sample)}"
        self.session_status.setText(status if self._connected or self.session_id else "")
        idle = self._connected and self.pending_request_id is None
        final = state == "finalized" and self.session_id is not None
        self.start_button.setText("结束本次会话" if final else "开始录音")
        self.start_button.setEnabled(idle and (state == "ready" or final))
        self.stop_button.setEnabled(idle and state == "recording" and self.session_id is not None)
        self.cancel_button.setEnabled(idle and state == "refining" and self._job_id is not None)
        self.connect_button.setEnabled(self.connect_client is not None and self._transport is None)

    @Slot()
    def _stop_recording(self):
        options = {"wait": False} if self._async_stop_supported else {}
        self._request("stop", session_id=self.session_id, **options)

    @Slot()
    def _start_or_complete(self):
        if self._state == "finalized":
            self._request("complete", session_id=self.session_id)
        else:
            self._request("start", **self.start_options)

    def _show_error(self, text):
        self.error_status.setText(text)
        self.error_status.setVisible(bool(text))

    @Slot()
    def _connect(self):
        self._show_error("")
        self.connection_status.setText("连接中…")
        self._transport = _Transport(self, self.connect_client)
        self._transport.connected.connect(self._on_connected)
        self._transport.received.connect(self._on_received)
        self._transport.failed.connect(self._on_failed)
        self._set_state(self._state)
        self._transport.open()

    @Slot(object)
    def _on_connected(self, transport):
        if transport is not self._transport:
            return
        self._connected = True
        self.connection_status.setText(f"{self.connection_name}  ·  已连接")
        if self.pending_request_id is not None:
            self._transport.requests.put((self._pending_operation, self.pending_request_id,
                                          self._pending_fields))
        elif self.session_id:
            self._request("resume", session_id=self.session_id,
                          last_applied_event_id=self._event_watermark or 0)
        self._set_state(self._state)

    @Slot(object, str)
    def _on_failed(self, transport, message):
        if transport is not self._transport:
            return
        self._transport = None
        self._connected = False
        self.connection_status.setText("已断开")
        self._show_error(message)
        self._set_state(self._state)

    def _request(self, operation, **fields):
        if not self._connected or self.pending_request_id:
            return
        self._show_error("")
        self.pending_request_id = str(uuid4())
        self._pending_operation = operation
        self._pending_fields = fields
        self._transport.requests.put((operation, self.pending_request_id, fields))
        self._set_state(self._state)

    @Slot(object, object, object)
    def _on_received(self, transport, frame, received_ns):
        if transport is not self._transport:
            return
        kind, data = frame
        if kind == MessageType.PCM:
            if self.pending_request_id and self._pending_operation == "start":
                self._early_audio.append(data)
            else:
                self._set_audio(data)
            return
        if kind == MessageType.RESPONSE:
            if self.pending_request_id is None or data.get("request_id") != self.pending_request_id:
                return
            self.pending_request_id = None
            if not data.get("ok"):
                code = data.get("code")
                self._show_error(ERRORS.get(code, "板端拒绝了本次操作"))
            elif self._pending_operation == "start" and data.get("session_id"):
                self.select_session(data["session_id"])
                for audio in self._early_audio:
                    self._set_audio(audio)
            elif self._pending_operation == "resume" and data.get("replay_available") is False:
                self._show_error("已重新连接板端。断线期间的转写可能缺失，当前保留本地已显示的原稿。")
            self._early_audio.clear()
        elif kind == MessageType.EVENT:
            if self.session_id is None or data.get("session_id") != self.session_id:
                return
            if data.get("type") == "transcript":
                self._apply(data, received_ns)
                return
            if data.get("type") == "capture_gap":
                self._show_capture_gap(data)
        if data.get("state") == "draining" and self._state in ("finalized", "completed", "faulted"):
            # A queued progress event or the async stop receipt can follow the last state.
            self._set_state(self._state)
            return
        if "refinement_job_id" in data:
            self._job_id = data["refinement_job_id"] or None
        if "async_stop_supported" in data:
            self._async_stop_supported = data["async_stop_supported"] is True
        if type(data.get("stop_sample")) is int and data["stop_sample"] >= 0:
            self._stop_sample = data["stop_sample"]
        if data.get("state") == "draining" and self._stop_sample is not None:
            processed = data.get("processed_sample")
            if type(processed) is int and 0 <= processed <= self._stop_sample:
                self._processed_sample = max(self._processed_sample or 0, processed)
        if data.get("error_code"):
            if data["error_code"] == "capture_storage_error":
                self._capture_storage_failed = True
            self._show_error(ERRORS.get(data["error_code"], "板端未能完成本次转写"))
        dropped = data.get("audio_preview_dropped_samples")
        if type(dropped) is int and dropped > self._preview_dropped_samples:
            self._preview_dropped_samples = dropped
            self._wave_dirty = True
        if data.get("state") in STATES:
            self._set_state(data["state"])
        else:
            self._set_state(self._state)

    def select_session(self, session_id):
        if self.session_id == session_id:
            return
        self._stop_playback()
        self.session_id = session_id
        self._event_watermark = None
        self._segments.clear()
        self._row_by_segment.clear()
        self._rows.clear()
        self.transcript_table.setRowCount(0)
        self.backend_status.setText("推理模式待确认")
        self._job_id = None
        self.first_text_received_ns = None
        self.first_text_visible_ns = None
        self._audio.clear()
        self._peaks.clear()
        self._gaps.clear()
        self._sample_end = 0
        self._gap_count = 0
        self._capture_gap_count = 0
        self._preview_dropped_samples = 0
        self._stop_sample = None
        self._processed_sample = None
        self._async_stop_supported = False
        self._capture_storage_failed = False
        self.capture_gap_status.clear()
        self.capture_gap_status.hide()
        self._wave_dirty = True
        self._show_selected()

    def apply(self, event: TranscriptEvent):
        self._event_ready.emit(dict(event), perf_counter_ns())

    @Slot(object, object)
    def _apply(self, event, received_ns):
        if self.session_id is None or event.get("session_id") != self.session_id:
            return
        segment = event.get("segment_id")
        try:
            validate_transcript_event(event, session_event_watermark=self._event_watermark)
        except ValueError:
            return
        previous = self._segments.get(segment)
        self._event_watermark = event["event_id"]
        if previous and (previous["is_final"] or event["revision"] <= previous["revision"]):
            return
        self._segments[segment] = event
        row = self._row_by_segment.get(segment)
        if row is None:
            row = len(self._rows)
            scrollbar = self.transcript_table.verticalScrollBar()
            follow_end = scrollbar.value() == scrollbar.maximum()
            self._rows.append(segment)
            self._row_by_segment[segment] = row
            self.transcript_table.insertRow(row)
            if follow_end:
                self.transcript_table.scrollToBottom()
        for column, value in enumerate((_time_at(event["sample_start"]),
                                       "已确认" if event["is_final"] else "识别中", event["text"])):
            item = self.transcript_table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.transcript_table.setItem(row, column, item)
            item.setText(value)
        mode = BACKENDS[event["backend"]]
        self.backend_status.setText(f"推理模式  ·  {mode}")
        self.backend_status.setToolTip(event["model_release"])
        if event["text"].strip() and self.first_text_received_ns is None:
            self.first_text_received_ns = received_ns
            self._first_text_row = row
        if self.transcript_table.currentRow() == row:
            self._show_selected()
        elif self.transcript_table.currentRow() < 0:
            self.transcript_table.selectRow(row)

    @Slot()
    def _record_first_visible(self):
        if self.first_text_received_ns is None or self.first_text_visible_ns is not None:
            return
        item = self.transcript_table.item(self._first_text_row, 2)
        if not item.text().strip() or not self.transcript_table.visualItemRect(item).intersects(
                self.transcript_table.viewport().rect()):
            return
        self.first_text_visible_ns = perf_counter_ns()
        LOG.info("first_text_visible session=%s received_ns=%d painted_ns=%d",
                 self.session_id, self.first_text_received_ns, self.first_text_visible_ns)

    def set_audio(self, block: AudioBlock):
        self._audio_ready.emit(block)

    def _show_capture_gap(self, data):
        start, end = data.get("sample_start"), data.get("sample_end")
        if type(start) is not int or start < 0:
            return
        self._capture_gap_count += 1
        extent = f"{_time_at(start)} 至 {_time_at(end)}" if type(end) is int and end > start else (
            f"{_time_at(start)} 起，范围未知")
        reason = CAPTURE_GAP_REASONS.get(data.get("reason"), "设备错误")
        self.capture_gap_status.setText(
            f"板端采集缺口 {self._capture_gap_count} 处  ·  最近 {extent}  ·  {reason}")
        self.capture_gap_status.show()

    @Slot(object)
    def _set_audio(self, block):
        if block.session_id != self.session_id or block.sample_count == 0:
            return
        end = block.sample_start + block.sample_count
        if end <= self._sample_end:
            return
        start = max(block.sample_start, self._sample_end)
        pcm = block.pcm_s16le[2 * (start - block.sample_start):]
        if start > self._sample_end:
            self._gaps.append((self._sample_end, start))
            self._gap_count += 1
        self._audio.append((start, end, pcm))
        for offset in range(0, end - start, 320):
            values = [sample for (sample,) in struct.iter_unpack("<h", pcm[offset * 2:(offset + 320) * 2])]
            self._peaks.append((start + offset, start + offset + len(values), min(values), max(values)))
        self._sample_end = end
        for spans, cutoff in ((self._audio, PREVIEW_SAMPLES), (self._peaks, WAVEFORM_SAMPLES),
                              (self._gaps, WAVEFORM_SAMPLES)):
            while spans and spans[0][1] <= end - cutoff:
                spans.popleft()
        self._wave_dirty = True

    @Slot()
    def refresh_waveform(self):
        if not self._wave_dirty:
            return
        self._wave_dirty = False
        self.waveform.sample_end = self._sample_end
        self.waveform.traces = list(self._peaks)
        self.waveform.gaps = list(self._gaps)
        self.waveform.update()
        status = "等待板端 PCM 音频"
        if self._peaks:
            _, _, lower, upper = self._peaks[-1]
            level = max(abs(lower), abs(upper)) * 100 // 32768
            begin = max(0, self._sample_end - WAVEFORM_SAMPLES)
            status = (f"{_time_at(begin)} 至 {_time_at(self._sample_end)}  ·  峰值 {level}%  ·  "
                      f"预览缺口 {self._gap_count} 处")
        if self._preview_dropped_samples:
            status += f"  ·  板端丢弃 {self._preview_dropped_samples:,} 个预览样本"
        self.waveform_status.setText(status)

    def seek(self, sample_index: int, sample_end: int | None = None) -> bytes:
        """Return PCM starting at the exact sample and ending before any preview gap."""
        if (type(sample_index) is not int or sample_index < 0
                or sample_index < self._sample_end - PREVIEW_SAMPLES):
            return b""
        if sample_end is None:
            sample_end = self._sample_end
        chunks = []
        position = sample_index
        for start, end, pcm in self._audio:
            if position >= end:
                continue
            if position < start or position >= sample_end:
                break
            stop = min(sample_end, end)
            chunks.append(pcm[(position - start) * 2:(stop - start) * 2])
            position = stop
        return b"".join(chunks)

    def _selected_segment(self):
        row = self.transcript_table.currentRow()
        return self._segments[self._rows[row]] if 0 <= row < len(self._rows) else None

    @Slot()
    def _show_selected(self):
        segment = self._selected_segment()
        text = segment["text"] if segment else ""
        if text != self.segment_text.toPlainText():
            self.segment_text.setPlainText(text)
        self.play_button.setEnabled(segment is not None)

    @Slot()
    def _play_selected(self):
        if self._buffer is not None:
            self._stop_playback()
            self.playback_status.setText("已停止试听")
            return
        segment = self._selected_segment()
        if segment is None:
            return
        start, end = segment["sample_start"], segment["sample_end"]
        pcm = self.seek(start, end)
        if not pcm:
            self.playback_status.setText("片段起点音频暂不可试听，可能已超出预览范围或存在缺口。")
            return
        if self._audio_output is None:
            fmt = QAudioFormat()
            fmt.setSampleRate(SAMPLE_RATE)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            self._audio_output = QAudioSink(fmt, self)
            self._audio_output.stateChanged.connect(self._playback_state_changed, Qt.ConnectionType.QueuedConnection)
        self._buffer = QBuffer(self)
        self._buffer.setData(pcm)
        self._buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        actual_end = start + len(pcm) // 2
        suffix = "，后续音频未收到" if actual_end < end else ""
        self.playback_status.setText(f"试听 {_time_at(start)} 至 {_time_at(actual_end)}{suffix}")
        self.play_button.setText("停止试听")
        self._audio_output.start(self._buffer)
        self._playback_state_changed(self._audio_output.state())

    def _playback_state_changed(self, status):
        if self._buffer is None or status != self._audio_output.state():
            return
        if status == QAudio.State.IdleState and self._buffer.atEnd():
            self._stop_playback()
            self.playback_status.setText("试听结束")
        elif status == QAudio.State.StoppedState and self._audio_output.error() != QAudio.Error.NoError:
            self._stop_playback()
            self.playback_status.setText("无法播放音频，请检查 Windows 音频输出设备。")

    def _stop_playback(self):
        if self._buffer is not None:
            buffer = self._buffer
            self._buffer = None
            self._audio_output.reset()
            buffer.close()
            buffer.deleteLater()
            self.play_button.setText("试听所选片段")

    def closeEvent(self, event):
        if self._transport:
            self._transport.close()
            self._transport = None
        self._stop_playback()
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description="FZ3B 实时语音识别客户端")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--host", help="板端 SSH 主机")
    source.add_argument("--loopback", action="store_true", help="连接本机回环服务或手动 SSH 转发")
    parser.add_argument("--port", type=int, default=22, help="SSH 端口")
    parser.add_argument("--username", help="SSH 用户名")
    parser.add_argument("--host-key", help="板端 SSH 公钥 SHA256 指纹")
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--identity-file", type=Path, help="SSH 私钥路径")
    identity.add_argument("--choose-key", action="store_true", help="打开窗口选择 SSH 私钥")
    parser.add_argument("--ask-password", action="store_true", help="输入 SSH 密码或私钥口令")
    parser.add_argument("--control-port", type=int, default=18765)
    parser.add_argument("--archive-port", type=int, default=18766)
    args = parser.parse_args(argv)
    if args.host and not (args.username and args.host_key):
        parser.error("SSH 连接需要 --username 和 --host-key")
    if args.loopback and (args.username or args.host_key or args.identity_file or args.choose_key
                          or args.ask_password):
        parser.error("回环连接不使用 SSH 凭据")

    logging.basicConfig(level=logging.INFO)
    app = QApplication([])
    key_path = args.identity_file
    if args.choose_key:
        filename, _ = QFileDialog.getOpenFileName(None, "选择 SSH 私钥", str(Path.home() / ".ssh"))
        if not filename:
            return 1
        key_path = Path(filename)
    password = None
    if args.ask_password:
        password, accepted = QInputDialog.getText(None, "SSH 身份验证",
                                                  "私钥口令" if key_path else "SSH 登录密码",
                                                  QLineEdit.EchoMode.Password)
        if not accepted:
            return 1

    def connect_client():
        if args.host:
            from paramiko import PKey

            private_key = None
            if key_path:
                passphrase = password.encode() if password else None
                private_key = PKey.from_path(key_path, passphrase=passphrase)
            return FzasClient.connect_ssh(args.host, args.username, expected_host_key=args.host_key,
                                          port=args.port, control_port=args.control_port,
                                          archive_port=args.archive_port, private_key=private_key,
                                          password=password if key_path is None else None)
        return FzasClient.connect_loopback(control_port=args.control_port, archive_port=args.archive_port)

    window = SessionView(connect_client=connect_client,
                         connection_name="本机回环" if args.loopback else "板端 SSH")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
