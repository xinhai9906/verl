"""Shared QAT quant-error probe recorder.

Records per-layer quantization vs original value MAE, aggregated by training
step, and flushes JSONL reports at step boundaries.  Used by both HiF8 and MXFP8
QAT linear layers without introducing a circular dependency on either.
"""

import atexit
import json
import logging
import os
import threading
from typing import Any, Optional

import torch

__all__ = [
    "QATProbeRecorder",
    "configure_qat_probe",
    "flush_qat_probe",
    "reset_qat_probe",
    "set_qat_probe_step",
    "qat_probe_step_context",
    "get_qat_probe_recorder",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# singleton recorder (module-private)
# ---------------------------------------------------------------------------

_QAT_PROBE_RECORDER: Optional["QATProbeRecorder"] = None
_QAT_PROBE_LOCK = threading.Lock()


def get_qat_probe_recorder() -> "QATProbeRecorder":
    """Return the process-wide singleton QATProbeRecorder (lazy init)."""
    global _QAT_PROBE_RECORDER
    if _QAT_PROBE_RECORDER is None:
        with _QAT_PROBE_LOCK:
            if _QAT_PROBE_RECORDER is None:
                _QAT_PROBE_RECORDER = QATProbeRecorder()
    return _QAT_PROBE_RECORDER


# ---------------------------------------------------------------------------
# public convenience helpers (match the MXFP8 API surface)
# ---------------------------------------------------------------------------


def configure_qat_probe(
    enabled: bool,
    output_path: Optional[str] = None,
    rank0_only: bool = True,
) -> None:
    """Enable / disable the quant-error probe and set the output path."""
    get_qat_probe_recorder().configure(
        enabled=enabled,
        output_path=output_path,
        rank0_only=rank0_only,
    )


def flush_qat_probe() -> None:
    """Force-flush accumulated per-step aggregates to log / file."""
    get_qat_probe_recorder().flush()


def reset_qat_probe() -> None:
    """Reset the probe recorder to its initial disabled state."""
    get_qat_probe_recorder().reset()


def set_qat_probe_step(step: Any) -> None:
    """Set the current training step so the recorder can detect step boundaries."""
    get_qat_probe_recorder().set_step(step)


class qat_probe_step_context:
    """Context manager that temporarily overrides the probe step."""

    def __init__(self, step: Any) -> None:
        self._step = step
        self._prev: Any = None

    def __enter__(self) -> "qat_probe_step_context":
        recorder = get_qat_probe_recorder()
        self._prev = recorder.current_step
        recorder.set_step(self._step)
        return self

    def __exit__(self, *args: Any) -> None:
        get_qat_probe_recorder().set_step(self._prev)


# ===========================================================================
# core recorder
# ===========================================================================


class QATProbeRecorder:
    """Thread-safe, step-aware quant-error aggregator + JSONL writer.

    Semantics
    ---------
    * Every call to :meth:`record` increments per-(step, …) accumulators.
    * When ``set_step`` detects a step change the previous step's aggregates are
      flushed (logged + optionally written as JSONL).
    * The singleton pattern via :func:`get_qat_probe_recorder` means every QAT
      layer in the process writes into the same recorder, producing one unified
      report per step.

    Output schema (one JSON object per aggregate key)::

        {
          "error_metric": "mae",
          "error_type": "weight" | "activation",
          "error_value": 0.00123,
          "layer_index": 12,
          "layer_type": "q_proj",
          "mode": "w8a16_hif8",
          "granularity": "per_channel",
          "group_size": 32,
          "rank": 0,
          "step": 100
        }
    """

    def __init__(self) -> None:
        self.enabled = False
        self.output_path: Optional[str] = None
        self.rank0_only = True
        self.current_step: Any = None
        self._aggregates: dict[tuple, dict[str, Any]] = {}
        self._last_recorded_step_key: Any = _SENTINEL
        self._fp: Optional[Any] = None  # TextIOWrapper
        self._lock = threading.Lock()
        self._atexit_registered = False

    # -- public API ----------------------------------------------------------

    def configure(
        self,
        enabled: bool,
        output_path: Optional[str] = None,
        rank0_only: bool = True,
    ) -> None:
        with self._lock:
            self._flush_locked()
            self._close_locked()
            self.enabled = enabled
            self.output_path = output_path if enabled else None
            self.rank0_only = rank0_only
            self.current_step = None
            self._aggregates.clear()
            self._last_recorded_step_key = _SENTINEL
            if enabled and output_path is not None and not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True

    def set_step(self, step: Any) -> None:
        """Set the current training step, flushing previous step data if changed.

        Flushing here (rather than lazily in :meth:`record`) ensures data is
        persisted immediately at step boundaries, so the final step is not
        lost when ``atexit`` fails to fire (e.g. process killed or file renamed
        before exit).
        """
        with self._lock:
            if self.current_step is not None and self.current_step != step:
                self._flush_locked()
            self.current_step = step

    def record(self, meta: dict, error_sum: float, element_count: int) -> None:
        """Accumulate one quant-error measurement.

        Parameters
        ----------
        meta:
            Dict with at least ``step``.  Other recommended keys:
            ``error_type``, ``layer_index``, ``layer_type``, ``mode``,
            ``granularity``, ``group_size``, ``rank``.
        error_sum:
            Sum of absolute differences ``|original - quantized|`` over all
            elements in this measurement.
        element_count:
            Total number of elements in this measurement.
        """
        if not self.enabled or not self._is_rank0():
            return
        step_raw = meta.get("step")
        if step_raw is None or element_count <= 0:
            return

        step_key = _freeze_value(step_raw)
        key = (
            step_key,
            meta.get("error_type"),
            meta.get("layer_index"),
            meta.get("layer_type"),
            meta.get("mode"),
            meta.get("granularity"),
            meta.get("group_size"),
            meta.get("rank"),
        )
        with self._lock:
            if self._last_recorded_step_key is not _SENTINEL and self._last_recorded_step_key != step_key:
                self._flush_locked()
            self._last_recorded_step_key = step_key
            agg = self._aggregates.setdefault(
                key,
                {"meta": meta, "error_sum": 0.0, "element_count": 0},
            )
            agg["error_sum"] += float(error_sum)
            agg["element_count"] += int(element_count)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def reset(self) -> None:
        with self._lock:
            self._flush_locked()
            self._close_locked()
            self.enabled = False
            self.output_path = None
            self.rank0_only = True
            self.current_step = None
            self._aggregates.clear()
            self._last_recorded_step_key = _SENTINEL

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            self._close_locked()

    # -- internals -----------------------------------------------------------

    def _is_rank0(self) -> bool:
        if not self.rank0_only:
            return True
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _ensure_open_locked(self) -> None:
        if self._fp is not None or self.output_path is None:
            return
        output_dir = os.path.dirname(os.path.abspath(self.output_path)) or "."
        os.makedirs(output_dir, exist_ok=True)
        self._fp = open(self.output_path, "a", encoding="utf-8")
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

    def _close_locked(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _flush_locked(self) -> None:
        if not self._aggregates:
            return

        records: list[dict] = []
        for agg in self._aggregates.values():
            element_count = agg["element_count"]
            if element_count <= 0:
                continue
            record = agg["meta"].copy()
            record["error_metric"] = "mae"
            record["error_value"] = agg["error_sum"] / element_count
            records.append(record)

        records.sort(key=_record_sort_key)
        for record in records:
            payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
            logger.warning("[QAT probe] %s", payload)

            if self.output_path is None:
                continue
            self._ensure_open_locked()
            if self._fp is not None:
                self._fp.write(payload + "\n")

        if self._fp is not None:
            self._fp.flush()
        self._aggregates.clear()
        self._last_recorded_step_key = _SENTINEL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _freeze_value(value: Any) -> Any:
    """Convert a step value into a hashable key."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            return int(value.item())
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value) if value else None
    return int(value)


def _record_sort_key(record: dict) -> tuple:
    return (
        json.dumps(record.get("step"), sort_keys=True),
        str(record.get("error_type")),
        record.get("layer_index") if record.get("layer_index") is not None else -1,
        str(record.get("layer_type")),
        str(record.get("rank")),
    )
