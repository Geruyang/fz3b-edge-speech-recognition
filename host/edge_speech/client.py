"""FZAS control and archive channels over a pinned SSH connection."""

import base64
from collections.abc import Mapping
import hashlib
from hmac import compare_digest
import json
import re
import socket
from threading import Lock
from time import monotonic
from uuid import UUID, uuid4

from .types import (
    HEADER,
    PCM_HEADER,
    AudioBlock,
    Frame,
    MessageType,
    decode_frame,
    decode_frame_header,
    encode_frame,
)


class ProtocolError(ConnectionError):
    """The channel was closed after receiving an invalid FZAS frame."""


class HostKeyMismatch(ConnectionError):
    """The server's key did not match its configured SHA-256 fingerprint."""


def frame_json(frame: Frame) -> dict:
    def reject_nonfinite(value):
        raise ValueError(f"invalid JSON constant {value}")

    try:
        value = json.loads(frame.payload.decode("utf-8"), parse_constant=reject_nonfinite)
    except (ValueError, RecursionError) as exc:
        raise ProtocolError("invalid JSON in FZAS frame") from exc
    if not isinstance(value, dict):
        raise ProtocolError("FZAS JSON payload must be an object")
    return value


def decode_audio(frame: Frame) -> AudioBlock:
    if frame.message_type != MessageType.PCM:
        raise ValueError("expected an FZAS PCM frame")
    session_id, first, count, flags = PCM_HEADER.unpack_from(frame.payload)
    return AudioBlock(str(UUID(bytes=session_id)), first, count, flags,
                      frame.payload[PCM_HEADER.size:])


class FzasChannel:
    """Synchronous byte stream. One reader and one concurrent writer are supported."""

    def __init__(self, stream, *, frame_timeout: float = 5):
        if frame_timeout <= 0:
            raise ValueError("frame_timeout must be positive")
        self.stream = stream
        self._frame_timeout = frame_timeout
        self._pending = bytearray()
        self._frame_started = None
        self._frame_end = None
        self._receive_seq = None
        self._send_seq = 0
        self._send_lock = Lock()

    def _check_frame_timeout(self):
        if (self._frame_started is not None
                and monotonic() - self._frame_started >= self._frame_timeout):
            raise ProtocolError("FZAS partial frame timed out")

    def _read_to(self, end: int) -> None:
        while len(self._pending) < end:
            self._check_frame_timeout()
            try:
                data = self.stream.recv(end - len(self._pending))
            except TimeoutError:
                self._check_frame_timeout()
                raise
            except OSError as exc:
                raise ConnectionError("FZAS read failed") from exc
            if not data:
                if self._pending:
                    raise ProtocolError("FZAS stream ended during a partial frame")
                raise ConnectionError("FZAS channel closed")
            if self._frame_started is None:
                self._frame_started = monotonic()
            self._pending.extend(data)
            self._check_frame_timeout()

    def receive(self) -> Frame:
        try:
            if self._frame_end is None:
                self._read_to(HEADER.size)
                header = decode_frame_header(self._pending)
                self._frame_end = HEADER.size + header.payload_bytes
            self._read_to(self._frame_end)
            frame = decode_frame(bytes(self._pending))
            if self._receive_seq is not None and frame.transport_seq <= self._receive_seq:
                raise ProtocolError("FZAS transport_seq did not advance")
        except ValueError as exc:
            self.close()
            raise ProtocolError(str(exc)) from exc
        except ConnectionError:
            self.close()
            raise

        self._pending.clear()
        self._frame_started = self._frame_end = None
        self._receive_seq = frame.transport_seq
        return frame

    def send(self, message_type: MessageType, payload: bytes, flags: int = 0) -> int:
        with self._send_lock:
            seq = self._send_seq + 1
            wire = encode_frame(Frame(message_type, seq, payload, flags))
            try:
                self.stream.sendall(wire)
            except OSError as exc:
                self.close()
                raise ConnectionError("FZAS write failed") from exc
            self._send_seq = seq
            return seq

    def close(self) -> None:
        try:
            self.stream.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.stream.close()


_CONTROL_TYPES = frozenset((MessageType.HELLO, MessageType.RESPONSE,
                            MessageType.EVENT, MessageType.HEARTBEAT))
_ARCHIVE_TYPES = frozenset((MessageType.HELLO, MessageType.PCM,
                            MessageType.FILE_CHUNK, MessageType.HEARTBEAT))


def _json_payload(value: Mapping) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _verify_host_key(key, expected_host_key: str) -> None:
    digest = hashlib.sha256(key.asbytes()).digest()
    actual = "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")
    if not compare_digest(expected_host_key, actual):
        raise HostKeyMismatch(f"SSH host key mismatch (received {actual})")


def _authenticate(ssh, paramiko, username, private_key, password):
    if private_key is not None:
        ssh.auth_publickey(username, private_key)
    elif password is not None:
        ssh.auth_password(username, password, fallback=False)
    else:
        agent = paramiko.Agent()
        try:
            for key in agent.get_keys():
                try:
                    ssh.auth_publickey(username, key)
                except paramiko.AuthenticationException:
                    continue
                if ssh.is_authenticated():
                    break
        finally:
            agent.close()
    if not ssh.is_authenticated():
        raise ConnectionError("SSH authentication failed")


class FzasClient:
    """Independent control and archive FZAS streams. Requests return their IDs."""

    def __init__(self, control: FzasChannel, archive: FzasChannel, ssh=None):
        self.control = control
        self.archive = archive
        self._ssh = ssh

    @classmethod
    def connect_loopback(cls, control_port: int = 18765, archive_port: int = 18766,
                         *, timeout: float = 5):
        control = socket.create_connection(("127.0.0.1", control_port), timeout=timeout)
        try:
            archive = socket.create_connection(("127.0.0.1", archive_port), timeout=timeout)
        except BaseException:
            control.close()
            raise
        return cls(FzasChannel(control), FzasChannel(archive))

    @classmethod
    def connect_ssh(cls, host: str, username: str, *, expected_host_key: str,
                    port: int = 22, control_port: int = 18765, archive_port: int = 18766,
                    private_key=None, password: str | None = None, timeout: float = 5):
        """Forward both loopback ports using a trusted SHA256 pin and in-memory credentials."""
        if not isinstance(expected_host_key, str) or not re.fullmatch(
                r"SHA256:[A-Za-z0-9+/]{43}", expected_host_key):
            raise ValueError("an OpenSSH SHA256 host fingerprint is required")
        if private_key is not None and password is not None:
            raise ValueError("choose either an SSH private key or a password")
        import paramiko

        wire = socket.create_connection((host, port), timeout=timeout)
        ssh = None
        try:
            ssh = paramiko.Transport(wire)
            ssh.banner_timeout = ssh.auth_timeout = timeout
            ssh.start_client(timeout=timeout)
            _verify_host_key(ssh.get_remote_server_key(), expected_host_key)
            _authenticate(ssh, paramiko, username, private_key, password)
            control = ssh.open_channel("direct-tcpip", ("127.0.0.1", control_port),
                                       ("127.0.0.1", 0), timeout=timeout)
            archive = ssh.open_channel("direct-tcpip", ("127.0.0.1", archive_port),
                                       ("127.0.0.1", 0), timeout=timeout)
            control.settimeout(timeout)
            archive.settimeout(timeout)
            return cls(FzasChannel(control), FzasChannel(archive), ssh)
        except BaseException:
            if ssh is not None:
                ssh.close()
            wire.close()
            raise

    def send_request(self, op: str, request_id: str | None = None, **fields) -> str:
        if not isinstance(op, str) or not op.strip():
            raise ValueError("a control operation is required")
        request_id = str(uuid4()) if request_id is None else request_id
        if not isinstance(request_id, str) or str(UUID(request_id)) != request_id:
            raise ValueError("request_id must be a canonical UUID")
        if "protocol" in fields:
            raise ValueError("protocol is assigned by the client")
        request = {"protocol": 1, "request_id": request_id, "op": op, **fields}
        self.control.send(MessageType.REQUEST, _json_payload(request))
        return request_id

    def request_stop(self, request_id: str, session_id: str) -> str:
        return self.send_request("stop", request_id=request_id, session_id=session_id)

    def send_archive_ack(self, fields: Mapping) -> int:
        return self.archive.send(MessageType.ACK, _json_payload(fields))

    @staticmethod
    def _receive(channel: FzasChannel, allowed: frozenset, name: str) -> Frame:
        frame = channel.receive()
        try:
            if frame.message_type not in allowed:
                raise ProtocolError(f"unexpected message type on FZAS {name} channel")
            if frame.message_type == MessageType.HELLO:
                hello = frame_json(frame)
                if hello.get("channel") != name or hello.get("protocol") != 1:
                    raise ProtocolError(f"invalid FZAS {name} HELLO")
            elif frame.message_type not in (MessageType.PCM, MessageType.FILE_CHUNK):
                frame_json(frame)
        except ProtocolError:
            channel.close()
            raise
        return frame

    def receive_control(self) -> Frame:
        return self._receive(self.control, _CONTROL_TYPES, "control")

    def receive_archive(self) -> Frame:
        return self._receive(self.archive, _ARCHIVE_TYPES, "archive")

    def close(self) -> None:
        # Closing the SSH transport wakes both forwarded readers and writers together.
        if self._ssh is not None:
            self._ssh.close()
        else:
            self.control.close()
            self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
