"""The warm worker: ``python -m dataventura.worker``.

Speaks the protocol in PROTOCOL.md: JSON-RPC 2.0, one message per line, on the stdin and stdout it
was started with; tables travel as Arrow IPC streams in POSIX shared memory. Anything the user's
code prints goes to stderr, so it can't corrupt the channel.
"""

from __future__ import annotations

import _thread
import importlib.util
import json
import os
import queue
import sys
import threading
import time
import traceback
from typing import Any

import pyarrow as pa

import dataventura as dv
from dataventura._shm import Segment

PROTOCOL = 0

# JSON-RPC error codes.
INVALID_PARAMS = -32602
METHOD_NOT_FOUND = -32601
USER_ERROR = -32000
CANCELLED = -32001


class Worker:
    def __init__(self, out: Any) -> None:
        self.out = out
        self.write_lock = threading.Lock()
        self.requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
        # The id of the request running user code, which a cancel message may interrupt.
        self.running: Any = None
        self.running_lock = threading.Lock()

    def send(self, msg: dict[str, Any]) -> None:
        line = json.dumps(msg, separators=(",", ":"))
        with self.write_lock:
            self.out.write(line + "\n")
            self.out.flush()

    def read(self, stdin: Any) -> None:
        """Reader thread: queues requests and handles cancel messages as they arrive."""
        for raw in stdin:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("method") == "cancel":
                with self.running_lock:
                    if self.running is not None and self.running == msg.get("params", {}).get("id"):
                        # Raises KeyboardInterrupt in the main thread at its next bytecode. Code in
                        # a long C call won't see it; the host stops the worker after a grace period.
                        _thread.interrupt_main()
                continue
            self.requests.put(msg)
        self.requests.put(None)

    def serve(self) -> None:
        self.send(
            {
                "jsonrpc": "2.0",
                "method": "ready",
                "params": {
                    "protocol": PROTOCOL,
                    "python": sys.version.split()[0],
                    "pyarrow": pa.__version__,
                    "pid": os.getpid(),
                },
            }
        )
        while True:
            try:
                msg = self.requests.get()
            except KeyboardInterrupt:
                continue  # a cancel that arrived just after its request finished
            if msg is None or msg.get("method") == "shutdown":
                return
            rid = msg.get("id")
            try:
                result = self.dispatch(msg.get("method"), msg.get("params") or {}, rid)
                self.send({"jsonrpc": "2.0", "id": rid, "result": result})
            except KeyboardInterrupt:
                self.send(error(rid, CANCELLED, "cancelled"))
            except RpcError as e:
                self.send(error(rid, e.code, e.message, e.data))
            except Exception as e:  # noqa: BLE001 - user code may raise anything
                self.send(error(rid, USER_ERROR, f"{type(e).__name__}: {e}", traceback.format_exc()))
            finally:
                with self.running_lock:
                    self.running = None

    def dispatch(self, method: str, params: dict[str, Any], rid: Any) -> Any:
        if method == "load":
            return load(params["path"])
        if method == "transform":
            return self.transform(params, rid)
        raise RpcError(METHOD_NOT_FOUND, f"unknown method {method!r}")

    def transform(self, p: dict[str, Any], rid: Any) -> dict[str, Any]:
        t = dv.registry().get(p["name"])
        if t is None:
            raise RpcError(INVALID_PARAMS, f"no transform named {p['name']!r}")
        timings: dict[str, float] = {}
        clock = time.perf_counter

        t0 = clock()
        seg = Segment(p["input"]["segment"], p["input"]["size"], create=False)
        t1 = clock()
        table = pa.ipc.open_stream(pa.py_buffer(seg.view)).read_all()
        t2 = clock()
        arg = to_input(table, t.input_type)
        t3 = clock()
        with self.running_lock:
            self.running = rid
        try:
            out = t.func(arg)
        finally:
            with self.running_lock:
                self.running = None
        t4 = clock()
        result = to_table(out, t.name)
        del arg, out, table
        t5 = clock()
        size, rows = write_result(result, p["output"]["segment"])
        del result
        released = seg.close()
        t6 = clock()

        timings = {
            "attach": t1 - t0,
            "decode": t2 - t1,
            "to_input": t3 - t2,
            "user": t4 - t3,
            "to_table": t5 - t4,
            "encode": t6 - t5,
        }
        return {
            "output": {"segment": p["output"]["segment"], "size": size},
            "rows": rows,
            "input_released": released,
            "timings": {k: v * 1e3 for k, v in timings.items()},
        }


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def error(rid: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": rid, "error": err}


def load(path: str) -> dict[str, Any]:
    """Imports a .py file; its decorators register transforms."""
    name = "dv_user_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RpcError(INVALID_PARAMS, f"can't import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return {
        "transforms": [
            {"name": t.name, "input_type": t.input_type, "columns": t.columns}
            for t in dv.registry().values()
        ]
    }


def to_input(table: pa.Table, kind: str) -> Any:
    if kind == "arrow":
        return table
    if kind == "polars":
        import polars as pl

        return pl.from_arrow(table, rechunk=False)
    if kind == "pandas":
        return table.to_pandas()
    if kind == "numpy":
        if table.num_columns != 1:
            raise RpcError(INVALID_PARAMS, "a numpy.ndarray transform takes exactly one column")
        return table.column(0).to_numpy()
    raise RpcError(INVALID_PARAMS, f"unknown input type {kind!r}")


def to_table(out: Any, name: str) -> pa.Table:
    """The function's return value as an Arrow table."""
    if isinstance(out, pa.Table):
        return out
    if isinstance(out, pa.RecordBatch):
        return pa.Table.from_batches([out])
    if isinstance(out, (pa.Array, pa.ChunkedArray)):
        return pa.table({name: out})
    module = type(out).__module__.split(".")[0]
    if module == "polars":
        import polars as pl

        if isinstance(out, pl.Series):
            return pa.table({out.name or name: out.to_arrow()})
        return out.to_arrow()
    if module == "pandas":
        import pandas as pd

        if isinstance(out, pd.Series):
            return pa.table({str(out.name) if out.name is not None else name: pa.array(out)})
        return pa.Table.from_pandas(out, preserve_index=False)
    if module == "numpy":
        return pa.table({name: pa.array(out)})
    raise RpcError(USER_ERROR, f"transform {name} returned {type(out).__name__}, not a table")


def write_result(table: pa.Table, name: str) -> tuple[int, int]:
    """Writes `table` as an Arrow IPC stream into a new segment named `name`, sized exactly: the
    first pass only counts bytes, the second writes them in place."""
    mock = pa.MockOutputStream()
    with pa.ipc.new_stream(mock, table.schema) as w:
        w.write_table(table)
    size = mock.size()
    seg = Segment(name, size, create=True)
    sink = pa.FixedSizeBufferWriter(pa.py_buffer(seg.view))
    with pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    sink.close()
    del sink
    seg.close()
    return size, table.num_rows


def main() -> None:
    # The protocol keeps the original stdout; fd 1 and sys.stdout now go to stderr, so prints from
    # user code and C libraries land in the log.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    w = Worker(out)
    threading.Thread(target=w.read, args=(sys.stdin,), daemon=True).start()
    w.serve()


if __name__ == "__main__":
    main()
