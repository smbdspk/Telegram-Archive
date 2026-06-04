"""Per-file parallel download engine for large media files.

Downloads a single file using multiple concurrent MTProto GetFileRequest
calls on borrowed senders to the target DC.  Only used for files above a
configurable size threshold; small files use the proven single-stream
``client.download_media()`` path.

Peak memory ≈ workers × part_size (default: 4 × 1 MB = 4 MB).

Private Telethon APIs used (isolated in this module):
    - ``client._borrow_sender(dc_id)``

These are stable across Telethon 1.37 – 1.42+ but could break in a
future major version.  If unavailable, ``PARALLEL_DOWNLOAD_AVAILABLE``
is ``False`` and the feature silently disables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe for required Telethon internals at import time.
# If anything is missing the module still loads but the feature flag is off.
# ---------------------------------------------------------------------------
try:
    from telethon import TelegramClient
    from telethon import utils as telethon_utils
    from telethon.errors import FloodWaitError
    from telethon.tl.functions.upload import GetFileRequest
    from telethon.tl.types import upload as upload_types

    if not hasattr(TelegramClient, "_borrow_sender"):
        raise AttributeError("TelegramClient._borrow_sender not found")

    PARALLEL_DOWNLOAD_AVAILABLE = True
except (ImportError, AttributeError) as _probe_err:  # pragma: no cover
    PARALLEL_DOWNLOAD_AVAILABLE = False
    logger.warning(
        "Parallel download unavailable: %s. Large files will use single-stream download.",
        _probe_err,
    )

# Telegram API maximum for GetFileRequest *limit* parameter.
_MAX_REQUEST_SIZE = 1024 * 1024  # 1 MB
_MIN_REQUEST_SIZE = 4096  # 4 KB – alignment requirement


# ---------------------------------------------------------------------------
# FloodBudget — shared flood-wait accounting across parallel workers
# ---------------------------------------------------------------------------
class FloodBudget:
    """Shared flood-wait budget across all parallel workers.

    Tracks cumulative retries and wait time against the caller's limits
    (``MAX_FLOOD_RETRIES`` / ``MAX_FLOOD_WAIT_SECONDS``).  When any
    worker hits ``FloodWaitError``, **all** sibling workers pause via an
    ``asyncio.Event`` until the mandatory wait completes.
    """

    def __init__(self, max_retries: int, max_wait_seconds: int) -> None:
        self.max_retries = max_retries
        self.max_wait_seconds = max_wait_seconds
        self._total_retries = 0
        self._total_wait_seconds = 0.0
        self._lock = asyncio.Lock()
        self._resume = asyncio.Event()
        self._resume.set()  # not paused initially
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    async def account_flood_wait(self, wait_seconds: float) -> bool:
        """Record a flood-wait event.

        Returns ``True`` if within budget (sleeps, then resumes workers).
        Returns ``False`` if budget exceeded (sets *cancelled* flag).
        """
        async with self._lock:
            if self._cancelled:
                return False
            self._total_retries += 1
            self._total_wait_seconds += wait_seconds
            if self._total_retries > self.max_retries or wait_seconds > self.max_wait_seconds:
                self._cancelled = True
                self._resume.set()  # wake waiters so they can exit
                return False
            # Pause all workers for the duration of the flood wait.
            self._resume.clear()

        jitter = random.uniform(0.5, 2.0)
        effective_wait = max(0.0, wait_seconds) + jitter
        logger.warning(
            "Parallel download: FloodWait %.1fs (+ %.1fs jitter), budget %d/%d retries, %.0fs cumulative wait",
            wait_seconds,
            jitter,
            self._total_retries,
            self.max_retries,
            self._total_wait_seconds,
        )
        await asyncio.sleep(effective_wait)
        self._resume.set()  # resume all workers
        return True

    async def wait_if_paused(self) -> None:
        """Block until any active flood wait completes."""
        await self._resume.wait()

    def cancel(self) -> None:
        """Signal cancellation to all workers."""
        self._cancelled = True
        self._resume.set()


# ---------------------------------------------------------------------------
# ByteRangeTracker — deterministic reassembly verification
# ---------------------------------------------------------------------------
class ByteRangeTracker:
    """Track written byte ranges to guarantee deterministic reassembly.

    Every range is recorded via :meth:`mark_written`; duplicates and
    overlaps raise immediately.  After all workers finish,
    :meth:`verify_complete` asserts that the union of ranges equals
    ``[0, total_size)`` — i.e. every byte was written exactly once.
    """

    def __init__(self, total_size: int) -> None:
        self.total_size = total_size
        self._ranges: list[tuple[int, int]] = []
        self._lock = asyncio.Lock()
        self._total_written = 0

    @property
    def total_written(self) -> int:
        return self._total_written

    async def mark_written(self, offset: int, length: int) -> None:
        """Record that bytes ``[offset, offset+length)`` have been written.

        Raises ``RuntimeError`` on overlap with any previously recorded range.
        """
        async with self._lock:
            end = offset + length
            for ex_start, ex_end in self._ranges:
                if offset < ex_end and end > ex_start:
                    raise RuntimeError(f"Overlapping write: [{offset}, {end}) conflicts with [{ex_start}, {ex_end})")
            self._ranges.append((offset, end))
            self._total_written += length

    def verify_complete(self) -> bool:
        """Return ``True`` iff all bytes ``[0, total_size)`` are covered."""
        if self.total_size == 0:
            return len(self._ranges) == 0
        if self._total_written != self.total_size:
            return False
        sorted_ranges = sorted(self._ranges)
        expected = 0
        for start, end in sorted_ranges:
            if start != expected:
                return False
            expected = end
        return expected == self.total_size


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
async def _download_chunk(
    sender,
    input_location,
    offset: int,
    chunk_size: int,
    flood_budget: FloodBudget,
    cancel_event: asyncio.Event,
) -> bytes | None:
    """Download a single chunk with flood-wait handling.

    Returns chunk bytes, or ``None`` if the download was cancelled.
    """
    while True:
        if cancel_event.is_set() or flood_budget.cancelled:
            return None
        try:
            result = await sender.send(
                GetFileRequest(
                    location=input_location,
                    offset=offset,
                    limit=chunk_size,
                )
            )
        except FloodWaitError as e:
            ok = await flood_budget.account_flood_wait(e.seconds)
            if not ok:
                cancel_event.set()
                raise
            continue  # retry after flood wait
        except Exception:
            cancel_event.set()
            raise

        # CDN redirects are rare for private archives but must be handled.
        if isinstance(result, upload_types.FileCdnRedirect):
            cancel_event.set()
            raise RuntimeError("CDN redirect during parallel download — file requires single-stream download")

        data = result.bytes
        if not data:
            cancel_event.set()
            raise RuntimeError(f"Empty response at offset {offset} (expected {chunk_size} bytes)")

        return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def download_file_parallel(
    client,
    message,
    output_path: str | os.PathLike,
    file_size: int,
    *,
    workers: int = 4,
    part_size: int = _MAX_REQUEST_SIZE,
    max_flood_retries: int = 5,
    max_flood_wait_seconds: int = 3600,
) -> str:
    """Download a file using parallel chunk transfers.

    Pre-allocates the output file, downloads chunks via concurrent
    ``GetFileRequest`` calls on borrowed senders, writes each chunk at
    its exact byte offset, and verifies complete byte coverage before
    returning.

    The entire operation is a single **transactional unit**: any chunk
    failure cancels siblings, cleans up the partial file, and re-raises
    the original exception for the caller's retry logic.

    Args:
        client:  ``TelegramClient`` instance.
        message: Telethon ``Message`` with media.
        output_path: Destination path for the downloaded file.
        file_size: Expected file size in bytes (from Telegram metadata).
        workers: Concurrent download workers (hard max 8).
        part_size: Bytes per ``GetFileRequest``.
            Capped at 1 MB (Telegram API limit), aligned to 4 096.
        max_flood_retries: Max flood-wait retries (shared budget).
        max_flood_wait_seconds: Max single flood-wait duration.

    Returns:
        *output_path* as a string on success.

    Raises:
        FloodWaitError: Flood-wait budget exceeded.
        RuntimeError: Byte-range verification failed / CDN redirect.
        ValueError: Message has no downloadable media.
    """
    if not PARALLEL_DOWNLOAD_AVAILABLE:
        raise RuntimeError("Parallel download not available — missing Telethon internals")

    if file_size <= 0:
        raise ValueError(f"Invalid file_size: {file_size}")

    # Clamp parameters
    workers = max(1, min(workers, 8))
    part_size = max(_MIN_REQUEST_SIZE, min(part_size, _MAX_REQUEST_SIZE))
    part_size = (part_size // _MIN_REQUEST_SIZE) * _MIN_REQUEST_SIZE  # 4 KB align

    # Extract InputFileLocation and DC from message media
    media = message.media
    if hasattr(media, "document") and media.document:
        loc_source = media.document
    elif hasattr(media, "photo") and media.photo:
        loc_source = media.photo
    else:
        raise ValueError("Message has no downloadable media (no document or photo)")

    dc_id, input_location = telethon_utils.get_input_location(loc_source)

    # Chunk layout
    total_parts = (file_size + part_size - 1) // part_size
    workers = min(workers, total_parts)

    logger.info(
        "Parallel download: %d bytes, %d parts × %d bytes, %d workers, DC %d",
        file_size,
        total_parts,
        part_size,
        workers,
        dc_id,
    )

    output_path_str = str(output_path)

    # Pre-allocate output file to exact expected size
    with open(output_path_str, "wb") as f:
        f.truncate(file_size)

    # Shared coordination
    flood_budget = FloodBudget(max_flood_retries, max_flood_wait_seconds)
    range_tracker = ByteRangeTracker(file_size)
    cancel_event = asyncio.Event()

    senders: list = []
    try:
        # Borrow senders for the target DC.
        # Multiple calls may return the same sender (request-pipelining)
        # or distinct connections — either way, concurrent send() works.
        for _ in range(workers):
            sender = await client._borrow_sender(dc_id)
            senders.append(sender)

        # Divide parts into contiguous ranges per worker
        parts_per_worker = total_parts // workers
        remainder = total_parts % workers

        async def _worker(worker_id: int, sender, start_part: int, end_part: int) -> None:
            """Download an assigned contiguous range of parts."""
            # Each worker opens its own fd → independent seek position
            with open(output_path_str, "r+b") as f:
                for part_idx in range(start_part, end_part):
                    if cancel_event.is_set() or flood_budget.cancelled:
                        return

                    await flood_budget.wait_if_paused()
                    if flood_budget.cancelled:
                        return

                    offset = part_idx * part_size
                    chunk_size = min(part_size, file_size - offset)

                    data = await _download_chunk(
                        sender,
                        input_location,
                        offset,
                        chunk_size,
                        flood_budget,
                        cancel_event,
                    )
                    if data is None:
                        return  # cancelled

                    # Write at exact byte offset
                    f.seek(offset)
                    f.write(data)
                    f.flush()

                    await range_tracker.mark_written(offset, len(data))

        # Launch workers with contiguous part assignments
        tasks: list[asyncio.Task] = []
        current_part = 0
        for i in range(workers):
            n_parts = parts_per_worker + (1 if i < remainder else 0)
            if n_parts > 0:
                tasks.append(
                    asyncio.create_task(
                        _worker(i, senders[i], current_part, current_part + n_parts),
                        name=f"parallel-dl-worker-{i}",
                    )
                )
                current_part += n_parts

        # Wait for all workers, collecting exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Re-raise the first real error (preserves original exception type
        # so the caller's retry/refresh logic handles it unchanged).
        for result in results:
            if isinstance(result, BaseException):
                raise result

        # Deterministic reassembly verification
        if not range_tracker.verify_complete():
            raise RuntimeError(
                f"Parallel download verification failed: "
                f"expected {file_size} bytes, wrote {range_tracker.total_written} bytes "
                f"across {len(range_tracker._ranges)} ranges"
            )

        logger.info("Parallel download verified: %d bytes complete", file_size)
        return output_path_str

    except BaseException:
        # Cancel any still-running workers
        cancel_event.set()
        # Clean up partial file
        try:
            if os.path.exists(output_path_str):
                os.remove(output_path_str)
                logger.debug("Cleaned up partial file: %s", output_path_str)
        except OSError:
            pass
        raise
