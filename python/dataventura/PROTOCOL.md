# Connected-runtime protocol, version 0

How the Data Ventura engine (the *host*) talks to a worker process that runs user code. Python's
worker is `python -m dataventura.worker`; any language with Arrow IPC support can implement one.
Version 0 is a draft and may change before 1.0.

## Transport

- The host starts the worker with pipes for stdin, stdout and stderr.
- **Control:** JSON-RPC 2.0, one JSON object per line, UTF-8. Requests go to the worker's stdin;
  replies and notifications come from its stdout. Nothing else may be written to stdout, so the
  worker redirects its own stdout (including fd 1) to stderr before running user code.
- **Logs:** whatever the worker writes to stderr. The host keeps the most recent part to explain
  failures.
- **Data:** Arrow IPC *streams* in POSIX shared memory (`shm_open`). Messages carry segment names
  and exact byte lengths, never table data.

## Segments

The host names every segment, including the ones the worker creates for results. Names are short
(macOS allows 31 bytes), start with `/`, and are unique for the host's lifetime. The host records
each name before handing it out and unlinks it once the data is mapped or no longer needed, so
segments left behind by a crash are released the next time the host starts. A worker must not
unlink segments.

- **Input:** the host creates a segment, writes the IPC stream, and passes `{segment, size}`. The
  worker maps it read-only (or read-write) and reads `size` bytes. The worker may decode it in place.
- **Output:** the host passes a fresh name. The worker creates the segment with
  `O_CREAT | O_EXCL`, sized to the IPC stream it writes, and replies with `{segment, size}`. The host
  maps it read-only, unlinks the name and decodes the stream in place.

## Messages

### `ready` (notification, worker → host)

Sent once, as soon as the worker can take requests.

```json
{"jsonrpc":"2.0","method":"ready","params":{"protocol":0,"python":"3.13.1","pyarrow":"26.0.1","pid":1234}}
```

A host refuses a worker whose `protocol` differs from its own.

### `load` (request)

Imports a source file. Its decorators register transforms. The reply lists every transform
registered so far.

```json
{"jsonrpc":"2.0","id":1,"method":"load","params":{"path":"/work/transforms.py"}}
{"jsonrpc":"2.0","id":1,"result":{"transforms":[{"name":"trip_speed","input_type":"arrow","columns":["startLon","startLat"]}]}}
```

`input_type` is one of `arrow`, `polars`, `pandas`, `numpy`. `columns` is the list of input
columns the transform reads, or `null` for all of them. The host sends only those columns.

### `transform` (request)

```json
{"jsonrpc":"2.0","id":2,"method":"transform","params":{
  "name":"trip_speed","generation":17,
  "input":{"segment":"/dv812.4","size":48001234},
  "output":{"segment":"/dv812.5"}}}
{"jsonrpc":"2.0","id":2,"result":{
  "output":{"segment":"/dv812.5","size":16000912},"rows":1000000,"input_released":true,
  "timings":{"attach":0.01,"decode":0.09,"to_input":0.0,"user":29.5,"to_table":0.01,"encode":1.8}}}
```

`timings` are in milliseconds. `user` is the time spent in the user's function. `input_released`
is false if user code kept a reference to the input, which keeps it mapped in the worker.

### `cancel` (notification, host → worker)

```json
{"jsonrpc":"2.0","method":"cancel","params":{"id":2}}
```

Asks the worker to abandon request `id` if it's still running. The worker replies to that request
with error `-32001`. If it hasn't replied within the grace period (2 s by default), the host stops
the process and starts a new one for the next request.

### `shutdown` (notification, host → worker)

The worker exits. If it's still running after 500 ms, the host stops it.

## Errors

| Code | Meaning |
|---|---|
| -32000 | The user's code raised. `message` is `Type: text`; `data` is the traceback. |
| -32001 | Cancelled. |
| -32601 | Unknown method. |
| -32602 | Invalid parameters, such as an unknown transform name. |

## Limits the host enforces

- **Memory:** the host samples the worker's physical footprint every few milliseconds and stops the
  process if it goes over its cap. By default the cap is a quarter of physical memory. The request
  fails with a message that names the cap.
- **Crashes:** a worker that exits is replaced on the next request, and the files it had loaded are
  loaded again.
- **Shared memory** counts against the engine's memory budget for as long as it's mapped by the
  host.
