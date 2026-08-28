#!/usr/bin/env python3
"""Receive reliable detection-triggered JPEG snapshots from the RK3588 sender.

The media format is intentionally separate from the standard H.265 RTP flow:
it uses a tiny UDP stop-and-wait protocol so a receiver restart resumes the
same .part file instead of discarding a costly high-resolution upload on a
low-bandwidth link.  Disk synchronization is batched for speed; the final CRC
prevents an incomplete or corrupt JPG from being published.
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple


MAGIC = b"RSNP"
VERSION = 1
HEADER = struct.Struct("!4sBBHIIIHHHHI")
START, DATA, END, RESUME, ACK, COMPLETE, ABORT = range(1, 8)
TYPE_NAMES = {
    START: "START", DATA: "DATA", END: "END", RESUME: "RESUME",
    ACK: "ACK", COMPLETE: "COMPLETE", ABORT: "ABORT",
}
CLASS_NAMES = ((1 << 0, "person"), (1 << 1, "car"), (1 << 2, "boat"),
               (1 << 3, "airplane"))
# On Windows, a blocking Winsock recvfrom() may not return when Ctrl+C is
# pressed.  Keep the wait bounded so Python gets a chance to raise
# KeyboardInterrupt and run the durable-store cleanup path.
RECEIVE_POLL_TIMEOUT_SECONDS = 0.25


@dataclass
class Packet:
    packet_type: int
    transfer_id: int
    total_bytes: int
    offset: int = 0
    payload: bytes = b""
    flags: int = 0
    width: int = 0
    height: int = 0
    class_mask: int = 0
    crc32: int = 0

    def encode(self) -> bytes:
        if self.packet_type not in TYPE_NAMES:
            raise ValueError("invalid packet type")
        if len(self.payload) > 0xFFFF or self.offset > self.total_bytes:
            raise ValueError("invalid packet payload or offset")
        if len(self.payload) > self.total_bytes - self.offset:
            raise ValueError("payload runs beyond declared snapshot size")
        return HEADER.pack(MAGIC, VERSION, self.packet_type, self.flags,
                           self.transfer_id, self.total_bytes, self.offset,
                           len(self.payload), self.width, self.height,
                           self.class_mask, self.crc32) + self.payload

    @classmethod
    def decode(cls, datagram: bytes) -> "Packet":
        if len(datagram) < HEADER.size:
            raise ValueError("datagram shorter than 32-byte snapshot header")
        (magic, version, packet_type, flags, transfer_id, total_bytes, offset,
         payload_length, width, height, class_mask, crc32) = HEADER.unpack_from(datagram)
        if magic != MAGIC or version != VERSION or packet_type not in TYPE_NAMES:
            raise ValueError("invalid snapshot magic, version, or packet type")
        if len(datagram) != HEADER.size + payload_length:
            raise ValueError("snapshot payload length does not match datagram")
        if offset > total_bytes or payload_length > total_bytes - offset:
            raise ValueError("snapshot offset runs outside declared file")
        return cls(packet_type, transfer_id, total_bytes, offset,
                   datagram[HEADER.size:], flags, width, height, class_mask, crc32)


def class_names(mask: int) -> str:
    names = [name for bit, name in CLASS_NAMES if mask & bit]
    return ",".join(names) if names else "unknown"


def file_crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as source:
        while True:
            block = source.read(64 * 1024)
            if not block:
                return crc & 0xFFFFFFFF
            crc = zlib.crc32(block, crc)


@dataclass
class TransferState:
    transfer_id: int
    total_bytes: int
    width: int
    height: int
    class_mask: int
    crc32: int
    part_path: Path
    final_path: Path
    expected_offset: int = 0
    completed: bool = False
    # Keep the part file open across sequential UDP chunks.  Reopening and
    # fsyncing it for every 900–1100 B packet is especially expensive on
    # Windows and can dominate the time once a faster link is used.
    sink: Optional[BinaryIO] = field(default=None, repr=False, compare=False)
    dirty_bytes: int = 0


class SnapshotStore:
    """Sequential JPEG writer with bounded, restart-safe sync batches.

    DATA packets are acknowledged immediately after they are written to the
    open file.  The file is fsynced every ``sync_every_bytes`` and also before
    START/END replies and normal receiver shutdown.  Thus normal resume is
    durable, while an abrupt power loss can at worst make the sender resend a
    small final batch; END's CRC still prevents a corrupt JPG from completing.
    """

    def __init__(self, output_dir: Path, verbose: bool = True,
                 sync_every_bytes: int = 32 * 1024):
        if sync_every_bytes < 0:
            raise ValueError("sync_every_bytes must be non-negative")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.states: Dict[Tuple[int, int], TransferState] = {}
        self.verbose = verbose
        self.sync_every_bytes = sync_every_bytes

    @staticmethod
    def _open_sink(state: TransferState) -> BinaryIO:
        if state.sink is None or state.sink.closed:
            # Unbuffered writes remove an extra Python user-space buffering
            # layer.  The OS may cache them until the next bounded fsync.
            state.sink = state.part_path.open("r+b", buffering=0)
        return state.sink

    def _sync(self, state: TransferState) -> None:
        if state.completed:
            return
        sink = self._open_sink(state)
        sink.flush()
        os.fsync(sink.fileno())
        state.dirty_bytes = 0

    def _close_sink(self, state: TransferState, sync: bool) -> None:
        if state.sink is None:
            return
        if sync and not state.completed:
            self._sync(state)
        sink = state.sink
        state.sink = None
        sink.close()

    def _reset_part(self, state: TransferState) -> None:
        self._close_sink(state, sync=False)
        with state.part_path.open("w+b", buffering=0) as sink:
            sink.truncate(0)
            sink.flush()
            os.fsync(sink.fileno())
        state.expected_offset = 0
        state.dirty_bytes = 0

    def _state(self, packet: Packet) -> TransferState:
        key = (packet.transfer_id, packet.crc32)
        state = self.states.get(key)
        if state is not None:
            if state.total_bytes != packet.total_bytes:
                raise ValueError("transfer id/CRC reused with a different total size")
            return state
        stem = (f"snapshot_{packet.transfer_id:08x}_{packet.crc32:08x}_"
                f"{packet.width}x{packet.height}")
        part_path = self.output_dir / (stem + ".jpg.part")
        final_path = self.output_dir / (stem + ".jpg")
        completed = final_path.exists()
        expected = packet.total_bytes if completed else 0
        if not completed and part_path.exists():
            expected = part_path.stat().st_size
            if expected > packet.total_bytes:
                with part_path.open("r+b") as sink:
                    sink.truncate(0)
                    sink.flush()
                    os.fsync(sink.fileno())
                expected = 0
        if not completed and not part_path.exists():
            part_path.touch()
        state = TransferState(packet.transfer_id, packet.total_bytes, packet.width,
                              packet.height, packet.class_mask, packet.crc32,
                              part_path, final_path, expected, completed)
        self.states[key] = state
        return state

    @staticmethod
    def _reply(packet: Packet, reply_type: int, offset: int) -> Packet:
        return Packet(reply_type, packet.transfer_id, packet.total_bytes, offset,
                      b"", 0, packet.width, packet.height, packet.class_mask,
                      packet.crc32)

    def handle(self, packet: Packet) -> Optional[Packet]:
        if packet.packet_type not in (START, DATA, END):
            return None
        state = self._state(packet)
        if packet.packet_type == START:
            # START is the resumption boundary.  Synchronize any in-memory
            # tail before publishing its offset to a retransmitting sender.
            self._sync(state)
            return self._reply(packet, RESUME, state.expected_offset)
        if packet.packet_type == DATA:
            if state.completed:
                return self._reply(packet, ACK, state.total_bytes)
            if packet.offset == state.expected_offset:
                sink = self._open_sink(state)
                sink.seek(state.expected_offset)
                written = sink.write(packet.payload)
                if written != len(packet.payload):
                    raise OSError("short snapshot file write")
                state.expected_offset += len(packet.payload)
                state.dirty_bytes += len(packet.payload)
                if (self.sync_every_bytes > 0 and
                        state.dirty_bytes >= self.sync_every_bytes):
                    self._sync(state)
            return self._reply(packet, ACK, state.expected_offset)
        if state.completed:
            return self._reply(packet, COMPLETE, state.total_bytes)
        if state.expected_offset != state.total_bytes:
            return self._reply(packet, RESUME, state.expected_offset)
        self._sync(state)
        # Windows cannot atomically rename an open .part file, and closing
        # here also gives file_crc32 a fresh on-disk view of the final batch.
        self._close_sink(state, sync=False)
        if file_crc32(state.part_path) != state.crc32:
            self._reset_part(state)
            return self._reply(packet, RESUME, 0)
        os.replace(state.part_path, state.final_path)
        state.completed = True
        if self.verbose:
            print(f"complete {state.final_path} {state.total_bytes} B "
                  f"{state.width}x{state.height} classes={class_names(state.class_mask)}",
                  flush=True)
        return self._reply(packet, COMPLETE, state.total_bytes)

    def close(self) -> None:
        """Flush incomplete parts before a normal Ctrl+C/process exit."""
        for state in self.states.values():
            try:
                self._close_sink(state, sync=not state.completed)
            except OSError as exc:
                print(f"snapshot receiver could not sync {state.part_path}: {exc}",
                      file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5008, help="UDP listen port (default: 5008)")
    parser.add_argument("--output", type=Path, default=Path("runs/snapshots"),
                        help="directory for final JPG files and resumable .part files")
    parser.add_argument("--sync-every-bytes", type=int, default=32 * 1024,
                        help="fsync each completed byte batch; 0 means only at START/END/exit "
                             "(default: 32768)")
    parser.add_argument("--quiet", action="store_true", help="suppress completed-file notices")
    return parser.parse_args()


def create_listener(port: int) -> socket.socket:
    """Create a UDP listener whose receive wait can be interrupted promptly."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(RECEIVE_POLL_TIMEOUT_SECONDS)
    return sock


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        print("--port must be in 1..65535", file=sys.stderr)
        return 2
    if args.sync_every_bytes < 0:
        print("--sync-every-bytes must be non-negative", file=sys.stderr)
        return 2
    store = SnapshotStore(args.output, verbose=not args.quiet,
                          sync_every_bytes=args.sync_every_bytes)
    sock = create_listener(args.port)
    print(f"Snapshot UDP listening on 0.0.0.0:{args.port}; output={args.output}; "
          f"sync_every={args.sync_every_bytes} B", flush=True)
    packet_count = 0
    payload_bytes = 0
    last_report = time.monotonic()
    try:
        while True:
            try:
                datagram, peer = sock.recvfrom(65535)
            except socket.timeout:
                # Polling is intentional: it makes Ctrl+C reliable even while
                # no sender packet is arriving on Windows.
                pass
            else:
                packet_count += 1
                try:
                    packet = Packet.decode(datagram)
                    payload_bytes += len(packet.payload)
                    reply = store.handle(packet)
                    if reply is not None:
                        sock.sendto(reply.encode(), peer)
                except (OSError, ValueError) as exc:
                    print(f"ignored snapshot datagram from {peer[0]}:{peer[1]}: {exc}", file=sys.stderr,
                          flush=True)
            now = time.monotonic()
            if now - last_report >= 5.0:
                elapsed = now - last_report
                if packet_count:
                    print(f"snapshot packets={packet_count} payload={payload_bytes} B "
                          f"rate={payload_bytes * 8.0 / elapsed / 1000.0:.1f} kbps "
                          f"active={len(store.states)}", flush=True)
                packet_count = 0
                payload_bytes = 0
                last_report = now
    except KeyboardInterrupt:
        print("\nSnapshot receiver stopped.", flush=True)
    finally:
        store.close()
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
