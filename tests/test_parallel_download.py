"""Tests for parallel download engine (src/parallel_download.py).

Covers:
- ByteRangeTracker: deterministic byte-range reassembly verification
- FloodBudget: shared flood-wait accounting across parallel workers
- download_file_parallel: end-to-end parallel download with error handling
- Config: parallel_download_* configuration properties
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telethon.errors import FloodWaitError

from src.parallel_download import (
    _MAX_REQUEST_SIZE,
    ByteRangeTracker,
    FloodBudget,
    download_file_parallel,
)


# ---------------------------------------------------------------------------
# ByteRangeTracker
# ---------------------------------------------------------------------------
class TestByteRangeTracker:
    """Test deterministic byte-range reassembly verification."""

    async def test_complete_coverage(self):
        """All ranges written contiguously → verify_complete() returns True."""
        tracker = ByteRangeTracker(1024)
        await tracker.mark_written(0, 512)
        await tracker.mark_written(512, 512)
        assert tracker.verify_complete()
        assert tracker.total_written == 1024

    async def test_gap_detection(self):
        """Missing range in the middle → verify_complete() returns False."""
        tracker = ByteRangeTracker(1024)
        await tracker.mark_written(0, 256)
        # Gap: 256..768 missing
        await tracker.mark_written(768, 256)
        assert not tracker.verify_complete()

    async def test_duplicate_write_raises(self):
        """Writing the same exact offset twice → RuntimeError."""
        tracker = ByteRangeTracker(1024)
        await tracker.mark_written(0, 512)
        with pytest.raises(RuntimeError, match="Overlapping write"):
            await tracker.mark_written(0, 512)

    async def test_wrong_offset_detection(self):
        """Chunk at wrong offset → verify_complete() fails due to gap."""
        tracker = ByteRangeTracker(1024)
        # Write at offset 100 instead of 0
        await tracker.mark_written(100, 512)
        await tracker.mark_written(612, 412)
        assert not tracker.verify_complete()

    async def test_overlap_detection(self):
        """Overlapping ranges → RuntimeError."""
        tracker = ByteRangeTracker(1024)
        await tracker.mark_written(0, 600)
        with pytest.raises(RuntimeError, match="Overlapping write"):
            await tracker.mark_written(500, 524)

    async def test_empty_file(self):
        """Zero-byte file → trivially complete with no ranges."""
        tracker = ByteRangeTracker(0)
        assert tracker.verify_complete()
        assert tracker.total_written == 0

    async def test_partial_coverage(self):
        """total_written != total_size → verify_complete() returns False."""
        tracker = ByteRangeTracker(1024)
        await tracker.mark_written(0, 500)
        assert not tracker.verify_complete()
        assert tracker.total_written == 500


# ---------------------------------------------------------------------------
# FloodBudget
# ---------------------------------------------------------------------------
@patch("src.parallel_download.random.uniform", return_value=1.0)
@patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
class TestFloodBudget:
    """Test shared flood-wait budget across parallel workers."""

    async def test_within_budget(self, mock_sleep, mock_uniform):
        """Retries within limit → returns True, not cancelled."""
        budget = FloodBudget(max_retries=5, max_wait_seconds=3600)
        result = await budget.account_flood_wait(10.0)
        assert result is True
        assert not budget.cancelled
        mock_sleep.assert_awaited_once()
        actual_sleep = mock_sleep.call_args[0][0]
        assert actual_sleep == pytest.approx(11.0, abs=0.1)

    async def test_retry_budget_exceeded(self, mock_sleep, mock_uniform):
        """Over retry limit → returns False, cancelled."""
        budget = FloodBudget(max_retries=2, max_wait_seconds=3600)
        await budget.account_flood_wait(1.0)
        await budget.account_flood_wait(1.0)
        # Third attempt exceeds max_retries
        result = await budget.account_flood_wait(1.0)
        assert result is False
        assert budget.cancelled

    async def test_wait_budget_exceeded(self, mock_sleep, mock_uniform):
        """Single wait exceeds max_wait_seconds → returns False, cancelled."""
        budget = FloodBudget(max_retries=5, max_wait_seconds=10)
        result = await budget.account_flood_wait(9999.0)
        assert result is False
        assert budget.cancelled
        # Should NOT have slept since budget was exceeded
        mock_sleep.assert_not_awaited()

    async def test_already_cancelled(self, mock_sleep, mock_uniform):
        """After cancel(), account_flood_wait returns False immediately."""
        budget = FloodBudget(max_retries=5, max_wait_seconds=3600)
        budget.cancel()
        result = await budget.account_flood_wait(1.0)
        assert result is False
        mock_sleep.assert_not_awaited()

    async def test_cancel_method(self, mock_sleep, mock_uniform):
        """cancel() sets cancelled flag and unblocks waiters."""
        budget = FloodBudget(max_retries=5, max_wait_seconds=3600)
        assert not budget.cancelled
        budget.cancel()
        assert budget.cancelled


# ---------------------------------------------------------------------------
# download_file_parallel
# ---------------------------------------------------------------------------
@patch("src.parallel_download.PARALLEL_DOWNLOAD_AVAILABLE", True)
class TestDownloadFileParallel:
    """Test end-to-end parallel download with mocked Telethon internals."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_download.bin")

        # Mock client
        self.client = MagicMock()
        self.mock_sender = MagicMock()
        self.mock_sender.send = AsyncMock()
        self.client._borrow_sender = AsyncMock(return_value=self.mock_sender)

        # Mock message with document media
        self.message = MagicMock()
        self.message.media = MagicMock()
        self.message.media.document = MagicMock()
        self.message.media.photo = None

        # Mock location
        self.mock_location = MagicMock()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _make_send_result(self, data: bytes) -> MagicMock:
        """Create a mock result for sender.send with spec=[] to avoid isinstance matches."""
        result = MagicMock(spec=[])
        result.bytes = data
        return result

    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_successful_download(self, mock_utils, mock_sleep):
        """Mock sender returns data chunks → file created with correct size."""
        file_size = 8192  # 2 parts at 4096 each
        part_size = 4096
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        chunk_a = b"A" * 4096
        chunk_b = b"B" * 4096
        self.mock_sender.send = AsyncMock(
            side_effect=[self._make_send_result(chunk_a), self._make_send_result(chunk_b)]
        )

        result = await download_file_parallel(
            self.client,
            self.message,
            self.output_path,
            file_size,
            workers=1,
            part_size=part_size,
        )

        assert result == self.output_path
        assert os.path.exists(self.output_path)
        with open(self.output_path, "rb") as f:
            content = f.read()
        assert len(content) == file_size
        assert content == chunk_a + chunk_b

    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_mid_transfer_file_reference_expired(self, mock_utils, mock_sleep):
        """FileReferenceExpiredError on 2nd chunk → cancelled, file cleaned, re-raised."""
        from telethon.errors import FileReferenceExpiredError

        file_size = 8192
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        self.mock_sender.send = AsyncMock(
            side_effect=[
                self._make_send_result(b"A" * 4096),
                FileReferenceExpiredError(request=None),
            ]
        )

        with pytest.raises(FileReferenceExpiredError):
            await download_file_parallel(
                self.client,
                self.message,
                self.output_path,
                file_size,
                workers=1,
                part_size=4096,
            )

        # Partial file should be cleaned up
        assert not os.path.exists(self.output_path)

    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_dropped_chunk_empty_response(self, mock_utils, mock_sleep):
        """sender.send returns empty bytes → RuntimeError, file cleaned."""
        file_size = 4096
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        empty_result = MagicMock(spec=[])
        empty_result.bytes = b""
        self.mock_sender.send = AsyncMock(return_value=empty_result)

        with pytest.raises(RuntimeError, match="Empty response"):
            await download_file_parallel(
                self.client,
                self.message,
                self.output_path,
                file_size,
                workers=1,
                part_size=4096,
            )

        assert not os.path.exists(self.output_path)

    @patch("src.parallel_download.random.uniform", return_value=1.0)
    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_flood_wait_budget_exceeded(self, mock_utils, mock_sleep, mock_uniform):
        """FloodWaitError with wait exceeding budget → re-raised, file cleaned."""
        file_size = 4096
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        self.mock_sender.send = AsyncMock(side_effect=FloodWaitError(request=None, capture=9999))

        with pytest.raises(FloodWaitError):
            await download_file_parallel(
                self.client,
                self.message,
                self.output_path,
                file_size,
                workers=1,
                part_size=4096,
                max_flood_wait_seconds=5,
            )

        assert not os.path.exists(self.output_path)

    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_connection_error_cleanup(self, mock_utils, mock_sleep):
        """ConnectionError → re-raised, file cleaned."""
        file_size = 4096
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        self.mock_sender.send = AsyncMock(side_effect=ConnectionError("lost"))

        with pytest.raises(ConnectionError, match="lost"):
            await download_file_parallel(
                self.client,
                self.message,
                self.output_path,
                file_size,
                workers=1,
                part_size=4096,
            )

        assert not os.path.exists(self.output_path)

    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_partial_file_cleanup(self, mock_utils, mock_sleep):
        """On any error, the output file is deleted."""
        file_size = 4096
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        self.mock_sender.send = AsyncMock(side_effect=OSError("disk full"))

        with pytest.raises(OSError):
            await download_file_parallel(
                self.client,
                self.message,
                self.output_path,
                file_size,
                workers=1,
                part_size=4096,
            )

        assert not os.path.exists(self.output_path)

    async def test_invalid_file_size(self):
        """file_size=0 → ValueError."""
        with pytest.raises(ValueError, match="Invalid file_size"):
            await download_file_parallel(self.client, self.message, self.output_path, file_size=0)

    async def test_no_media_raises(self):
        """Message media has no document or photo → ValueError."""
        self.message.media.document = None
        self.message.media.photo = None

        with patch("src.parallel_download.telethon_utils") as mock_utils:
            mock_utils.get_input_location.return_value = (2, self.mock_location)
            with pytest.raises(ValueError, match="no downloadable media"):
                await download_file_parallel(
                    self.client,
                    self.message,
                    self.output_path,
                    file_size=4096,
                )

    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_workers_capped_at_8(self, mock_utils, mock_sleep):
        """workers=20 → internally capped to 8."""
        file_size = 8 * _MAX_REQUEST_SIZE  # 8 parts, so all 8 workers get work
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        self.mock_sender.send = AsyncMock(return_value=self._make_send_result(b"X" * _MAX_REQUEST_SIZE))

        result = await download_file_parallel(
            self.client,
            self.message,
            self.output_path,
            file_size,
            workers=20,
            part_size=_MAX_REQUEST_SIZE,
        )

        # _borrow_sender should be called at most 8 times (capped)
        assert self.client._borrow_sender.await_count <= 8
        assert result == self.output_path

    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_part_size_capped_at_1mb(self, mock_utils, mock_sleep):
        """part_size=2*1024*1024 → capped to 1 MB (_MAX_REQUEST_SIZE)."""
        file_size = _MAX_REQUEST_SIZE
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        self.mock_sender.send = AsyncMock(return_value=self._make_send_result(b"X" * _MAX_REQUEST_SIZE))

        result = await download_file_parallel(
            self.client,
            self.message,
            self.output_path,
            file_size,
            workers=1,
            part_size=2 * 1024 * 1024,  # 2 MB, should be capped
        )

        assert result == self.output_path
        # Should have made exactly 1 request (1 MB file / 1 MB part = 1 part)
        assert self.mock_sender.send.await_count == 1

    @patch("src.parallel_download.asyncio.sleep", new_callable=AsyncMock)
    @patch("src.parallel_download.telethon_utils")
    async def test_part_size_aligned_to_4k(self, mock_utils, mock_sleep):
        """part_size=5000 → aligned down to 4096 (_MIN_REQUEST_SIZE)."""
        file_size = 4096
        mock_utils.get_input_location.return_value = (2, self.mock_location)

        self.mock_sender.send = AsyncMock(return_value=self._make_send_result(b"X" * 4096))

        result = await download_file_parallel(
            self.client,
            self.message,
            self.output_path,
            file_size,
            workers=1,
            part_size=5000,  # Not 4K aligned, should become 4096
        )

        assert result == self.output_path
        # Check the GetFileRequest limit parameter used
        call_args = self.mock_sender.send.call_args[0][0]
        assert call_args.limit == 4096


# ---------------------------------------------------------------------------
# Config parallel download properties
# ---------------------------------------------------------------------------
class TestParallelDownloadConfig(unittest.TestCase):
    """Test Config class parallel download configuration properties."""

    @patch("os.makedirs")
    @patch.dict(os.environ, {"BACKUP_PATH": "/tmp/test"}, clear=True)
    def test_default_disabled(self, mock_makedirs):
        """No env vars → parallel_download_enabled=False, workers=4, part_size_kb=1024."""
        from src.config import Config

        config = Config()
        self.assertFalse(config.parallel_download_enabled)
        self.assertEqual(config.parallel_download_workers, 4)
        self.assertEqual(config.parallel_download_part_size_kb, 1024)

    @patch("os.makedirs")
    @patch.dict(
        os.environ,
        {"BACKUP_PATH": "/tmp/test", "PARALLEL_DOWNLOAD_ENABLED": "true"},
        clear=True,
    )
    def test_enabled(self, mock_makedirs):
        """PARALLEL_DOWNLOAD_ENABLED=true → True."""
        from src.config import Config

        config = Config()
        self.assertTrue(config.parallel_download_enabled)

    @patch("os.makedirs")
    @patch.dict(
        os.environ,
        {"BACKUP_PATH": "/tmp/test", "PARALLEL_DOWNLOAD_WORKERS": "20"},
        clear=True,
    )
    def test_workers_hard_cap(self, mock_makedirs):
        """PARALLEL_DOWNLOAD_WORKERS=20 → capped to 8."""
        from src.config import Config

        config = Config()
        self.assertEqual(config.parallel_download_workers, 8)

    @patch("os.makedirs")
    @patch.dict(
        os.environ,
        {"BACKUP_PATH": "/tmp/test", "PARALLEL_DOWNLOAD_PART_SIZE_KB": "2048"},
        clear=True,
    )
    def test_part_size_cap(self, mock_makedirs):
        """PARALLEL_DOWNLOAD_PART_SIZE_KB=2048 → capped to 1024."""
        from src.config import Config

        config = Config()
        self.assertEqual(config.parallel_download_part_size_kb, 1024)

    @patch("os.makedirs")
    @patch.dict(
        os.environ,
        {"BACKUP_PATH": "/tmp/test", "PARALLEL_DOWNLOAD_MIN_SIZE_MB": "15"},
        clear=True,
    )
    def test_min_size_bytes(self, mock_makedirs):
        """min_size_mb=15 → get_parallel_download_min_size_bytes() == 15*1024*1024."""
        from src.config import Config

        config = Config()
        self.assertEqual(config.parallel_download_min_size_mb, 15)
        self.assertEqual(config.get_parallel_download_min_size_bytes(), 15 * 1024 * 1024)

    @patch("os.makedirs")
    @patch.dict(
        os.environ,
        {
            "BACKUP_PATH": "/tmp/test",
            "PARALLEL_DOWNLOAD_ENABLED": "true",
            "PARALLEL_DOWNLOAD_MIN_SIZE_MB": "25",
            "PARALLEL_DOWNLOAD_WORKERS": "6",
            "PARALLEL_DOWNLOAD_PART_SIZE_KB": "512",
        },
        clear=True,
    )
    def test_custom_values(self, mock_makedirs):
        """All custom values parsed correctly."""
        from src.config import Config

        config = Config()
        self.assertTrue(config.parallel_download_enabled)
        self.assertEqual(config.parallel_download_min_size_mb, 25)
        self.assertEqual(config.parallel_download_workers, 6)
        self.assertEqual(config.parallel_download_part_size_kb, 512)
        self.assertEqual(config.get_parallel_download_min_size_bytes(), 25 * 1024 * 1024)
