"""Tests for src.cleanup_media — orphan media blob detection and removal."""

import os
import unittest
from unittest.mock import MagicMock

import pytest

from src.cleanup_media import (
    _collect_dangling_symlinks,
    _collect_shared_blobs,
    _delete_dangling_sync,
    _delete_orphans_sync,
    _purge_chat_media_sync,
    clean_orphan_media,
    purge_skipped_chat_media,
)


class TestCollectSharedBlobs(unittest.TestCase):
    """Test _collect_shared_blobs filesystem scanner."""

    def test_empty_shared_dir(self, tmp_path=None):
        """Empty _shared/ returns empty dict."""
        if tmp_path is None:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                shared = os.path.join(td, "_shared")
                os.makedirs(shared)
                result = _collect_shared_blobs(shared)
                assert result == {}
            return

    def test_nonexistent_dir(self):
        """Non-existent directory returns empty dict."""
        result = _collect_shared_blobs("/nonexistent/path/_shared")
        assert result == {}

    def test_flat_layout(self, tmp_path=None):
        """Pre-sharding flat files are found."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            os.makedirs(shared)
            # Create flat files
            f1 = os.path.join(shared, "abc123_photo.jpg")
            with open(f1, "wb") as f:
                f.write(b"x" * 100)
            f2 = os.path.join(shared, "def456_video.mp4")
            with open(f2, "wb") as f:
                f.write(b"y" * 200)

            result = _collect_shared_blobs(shared)
            assert len(result) == 2
            assert os.path.realpath(f1) in result
            assert result[os.path.realpath(f1)] == 100
            assert result[os.path.realpath(f2)] == 200

    def test_sharded_layout(self):
        """Sharded files under <hash[:2]>/ buckets are found."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            bucket = os.path.join(shared, "ab")
            os.makedirs(bucket)
            f1 = os.path.join(bucket, "abc123_doc.pdf")
            with open(f1, "wb") as f:
                f.write(b"z" * 50)

            result = _collect_shared_blobs(shared)
            assert len(result) == 1
            assert result[os.path.realpath(f1)] == 50

    def test_mixed_layout(self):
        """Both flat and sharded files are found together."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            os.makedirs(shared)

            # Flat file
            flat = os.path.join(shared, "flat_file.jpg")
            with open(flat, "wb") as f:
                f.write(b"a" * 10)

            # Sharded file
            bucket = os.path.join(shared, "ff")
            os.makedirs(bucket)
            sharded = os.path.join(bucket, "ff123_sharded.mp4")
            with open(sharded, "wb") as f:
                f.write(b"b" * 20)

            result = _collect_shared_blobs(shared)
            assert len(result) == 2

    def test_skips_non_2char_subdirs(self):
        """Subdirectories that aren't 2-char shard buckets are skipped."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            os.makedirs(shared)

            # Non-bucket subdir
            other_dir = os.path.join(shared, "other")
            os.makedirs(other_dir)
            ignored = os.path.join(other_dir, "should_be_ignored.txt")
            with open(ignored, "w") as f:
                f.write("ignored")

            # Valid bucket
            bucket = os.path.join(shared, "ab")
            os.makedirs(bucket)
            found = os.path.join(bucket, "found.txt")
            with open(found, "w") as f:
                f.write("found")

            result = _collect_shared_blobs(shared)
            assert len(result) == 1
            assert os.path.realpath(found) in result

    def test_skips_marker_files(self):
        """Marker files (like .repaired-175-v2) are regular files and collected."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            os.makedirs(shared)
            marker = os.path.join(shared, ".repaired-175-v2")
            with open(marker, "w") as f:
                f.write("done")

            result = _collect_shared_blobs(shared)
            # Marker is a flat file — it will be collected as a blob
            # The cleanup logic's orphan detection will handle it correctly
            # since no DB record references it
            assert len(result) == 1


class TestCollectDanglingSymlinks(unittest.TestCase):
    """Test _collect_dangling_symlinks."""

    def test_no_dangling(self):
        """No dangling symlinks returns empty list."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media = td
            shared = os.path.join(media, "_shared", "ab")
            os.makedirs(shared)
            blob = os.path.join(shared, "blob.jpg")
            with open(blob, "wb") as f:
                f.write(b"x")

            chat_dir = os.path.join(media, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "blob.jpg")
            try:
                os.symlink(blob, link)
            except OSError:
                pytest.skip("symlinks not supported")

            result = _collect_dangling_symlinks(media)
            assert result == []

    def test_dangling_detected(self):
        """Dangling symlinks are detected."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media = td
            chat_dir = os.path.join(media, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "missing.jpg")
            try:
                os.symlink("/nonexistent/target", link)
            except OSError:
                pytest.skip("symlinks not supported")

            result = _collect_dangling_symlinks(media)
            assert len(result) == 1
            assert result[0] == link

    def test_skips_shared_and_avatars(self):
        """_shared/ and avatars/ directories are skipped."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media = td
            for skip_dir in ("_shared", "avatars"):
                d = os.path.join(media, skip_dir)
                os.makedirs(d)
                link = os.path.join(d, "dangling.jpg")
                try:
                    os.symlink("/nonexistent", link)
                except OSError:
                    pytest.skip("symlinks not supported")

            result = _collect_dangling_symlinks(media)
            assert result == []

    def test_nonexistent_media_path(self):
        """Non-existent media path returns empty list."""
        result = _collect_dangling_symlinks("/nonexistent/path")
        assert result == []


class TestDeleteOrphansSync(unittest.TestCase):
    """Test _delete_orphans_sync."""

    def test_deletes_orphan_files(self):
        """Orphan files are deleted and bytes counted."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            os.makedirs(shared)
            orphan = os.path.join(shared, "orphan.jpg")
            with open(orphan, "wb") as f:
                f.write(b"x" * 1000)

            deleted, freed, errors = _delete_orphans_sync([orphan], shared)
            assert deleted == 1
            assert freed == 1000
            assert errors == 0
            assert not os.path.exists(orphan)

    def test_cleans_empty_shard_dirs(self):
        """Empty shard bucket directories are removed after deletion."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            bucket = os.path.join(shared, "ab")
            os.makedirs(bucket)
            orphan = os.path.join(bucket, "orphan.jpg")
            with open(orphan, "wb") as f:
                f.write(b"x")

            _delete_orphans_sync([orphan], shared)
            assert not os.path.exists(bucket)

    def test_keeps_nonempty_shard_dirs(self):
        """Shard dirs with remaining files are not removed."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            bucket = os.path.join(shared, "ab")
            os.makedirs(bucket)
            orphan = os.path.join(bucket, "orphan.jpg")
            keeper = os.path.join(bucket, "keeper.jpg")
            with open(orphan, "wb") as f:
                f.write(b"x")
            with open(keeper, "wb") as f:
                f.write(b"y")

            _delete_orphans_sync([orphan], shared)
            assert os.path.exists(bucket)
            assert os.path.exists(keeper)

    def test_handles_already_deleted(self):
        """Already-deleted file (TOCTOU) is a no-op, not an error."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            shared = os.path.join(td, "_shared")
            os.makedirs(shared)
            deleted, freed, errors = _delete_orphans_sync(["/nonexistent/file.jpg"], shared)
            assert deleted == 0
            assert freed == 0
            assert errors == 0


class TestDeleteDanglingSync(unittest.TestCase):
    """Test _delete_dangling_sync."""

    def test_removes_dangling_symlinks(self):
        """Dangling symlinks are removed."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            link = os.path.join(td, "dangling.jpg")
            try:
                os.symlink("/nonexistent", link)
            except OSError:
                pytest.skip("symlinks not supported")

            deleted, errors = _delete_dangling_sync([link])
            assert deleted == 1
            assert errors == 0
            assert not os.path.lexists(link)

    def test_handles_already_removed(self):
        """Already-removed symlink is a no-op."""
        deleted, errors = _delete_dangling_sync(["/nonexistent/link"])
        assert deleted == 0
        assert errors == 0


class TestCleanOrphanMedia:
    """Integration tests for the main clean_orphan_media function."""

    def _make_mock_db(self, file_paths: list[str]) -> MagicMock:
        """Create a mock DB that yields the given file_paths in one batch."""
        db = MagicMock()

        async def _iter_paths(batch_size=5000):
            if file_paths:
                yield file_paths

        db.iter_all_media_file_paths = _iter_paths
        return db

    async def test_empty_shared_dir(self):
        """Empty _shared/ returns zero counts."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared")
            os.makedirs(shared)

            db = self._make_mock_db([])
            result = await clean_orphan_media(media_path, db)

            assert result["total_blobs"] == 0
            assert result["orphan_blobs"] == 0
            assert result["deleted_blobs"] == 0

    async def test_all_referenced(self):
        """No orphans when all blobs are referenced."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared", "ab")
            os.makedirs(shared)
            blob = os.path.join(shared, "abc_photo.jpg")
            with open(blob, "wb") as f:
                f.write(b"x" * 100)

            # DB references this blob directly
            db = self._make_mock_db([blob])
            result = await clean_orphan_media(media_path, db)

            assert result["total_blobs"] == 1
            assert result["referenced_blobs"] == 1
            assert result["orphan_blobs"] == 0

    async def test_all_referenced_via_symlink(self):
        """Symlink → blob is correctly resolved as referenced."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared", "ab")
            os.makedirs(shared)
            blob = os.path.join(shared, "abc_photo.jpg")
            with open(blob, "wb") as f:
                f.write(b"x" * 100)

            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "abc_photo.jpg")
            try:
                os.symlink(blob, link)
            except OSError:
                pytest.skip("symlinks not supported")

            # DB references the symlink path, not the blob
            db = self._make_mock_db([link])
            result = await clean_orphan_media(media_path, db)

            assert result["total_blobs"] == 1
            assert result["referenced_blobs"] == 1
            assert result["orphan_blobs"] == 0

    async def test_orphan_detected_dry_run(self):
        """Orphan blob is detected but NOT deleted in dry-run."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared")
            os.makedirs(shared)
            orphan = os.path.join(shared, "orphan.jpg")
            with open(orphan, "wb") as f:
                f.write(b"x" * 500)

            db = self._make_mock_db([])
            result = await clean_orphan_media(media_path, db, delete=False)

            assert result["orphan_blobs"] == 1
            assert result["orphan_bytes"] == 500
            assert result["deleted_blobs"] == 0
            assert result["freed_bytes"] == 0
            # File should still exist
            assert os.path.exists(orphan)

    async def test_orphan_deleted(self):
        """Orphan blob is deleted when delete=True."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared")
            os.makedirs(shared)
            orphan = os.path.join(shared, "orphan.jpg")
            with open(orphan, "wb") as f:
                f.write(b"x" * 500)

            db = self._make_mock_db([])
            result = await clean_orphan_media(media_path, db, delete=True)

            assert result["orphan_blobs"] == 1
            assert result["deleted_blobs"] == 1
            assert result["freed_bytes"] == 500
            assert not os.path.exists(orphan)

    async def test_referenced_not_deleted(self):
        """Referenced blobs are NOT deleted even with delete=True."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared", "ab")
            os.makedirs(shared)

            referenced = os.path.join(shared, "referenced.jpg")
            with open(referenced, "wb") as f:
                f.write(b"keep")

            orphan = os.path.join(shared, "orphan.jpg")
            with open(orphan, "wb") as f:
                f.write(b"delete_me")

            db = self._make_mock_db([referenced])
            result = await clean_orphan_media(media_path, db, delete=True)

            assert result["total_blobs"] == 2
            assert result["referenced_blobs"] == 1
            assert result["orphan_blobs"] == 1
            assert result["deleted_blobs"] == 1
            assert os.path.exists(referenced)
            assert not os.path.exists(orphan)

    async def test_dangling_symlinks_not_reported_by_default(self):
        """Dangling symlinks are NOT reported unless include_dangling=True."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared")
            os.makedirs(shared)

            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            try:
                os.symlink("/nonexistent", os.path.join(chat_dir, "dangling.jpg"))
            except OSError:
                pytest.skip("symlinks not supported")

            db = self._make_mock_db([])
            result = await clean_orphan_media(media_path, db)

            assert "dangling_symlinks" not in result

    async def test_dangling_symlinks_with_flag(self):
        """Dangling symlinks are reported with include_dangling=True."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared")
            os.makedirs(shared)

            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "dangling.jpg")
            try:
                os.symlink("/nonexistent", link)
            except OSError:
                pytest.skip("symlinks not supported")

            db = self._make_mock_db([])
            result = await clean_orphan_media(media_path, db, delete=False, include_dangling=True)

            assert result["dangling_symlinks"] == 1
            assert result["deleted_dangling"] == 0
            assert os.path.lexists(link)

    async def test_dangling_symlinks_deleted(self):
        """Dangling symlinks are removed with delete=True + include_dangling=True."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared")
            os.makedirs(shared)

            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "dangling.jpg")
            try:
                os.symlink("/nonexistent", link)
            except OSError:
                pytest.skip("symlinks not supported")

            db = self._make_mock_db([])
            result = await clean_orphan_media(media_path, db, delete=True, include_dangling=True)

            assert result["dangling_symlinks"] == 1
            assert result["deleted_dangling"] == 1
            assert not os.path.lexists(link)

    async def test_no_shared_dir(self):
        """No _shared/ directory returns zero counts gracefully."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            db = self._make_mock_db([])
            result = await clean_orphan_media(td, db)

            assert result["total_blobs"] == 0
            assert result["orphan_blobs"] == 0

    async def test_multiple_orphans_in_sharded_layout(self):
        """Multiple orphans across different shard buckets."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared")

            for bucket in ("aa", "bb", "cc"):
                bucket_dir = os.path.join(shared, bucket)
                os.makedirs(bucket_dir)
                orphan = os.path.join(bucket_dir, f"orphan_{bucket}.jpg")
                with open(orphan, "wb") as f:
                    f.write(b"x" * 100)

            db = self._make_mock_db([])
            result = await clean_orphan_media(media_path, db, delete=True)

            assert result["orphan_blobs"] == 3
            assert result["deleted_blobs"] == 3
            assert result["freed_bytes"] == 300
            # All empty shard dirs should be cleaned
            for bucket in ("aa", "bb", "cc"):
                assert not os.path.exists(os.path.join(shared, bucket))


class TestPurgeChatMediaSync(unittest.TestCase):
    """Test _purge_chat_media_sync filesystem helper."""

    def test_removes_real_files(self):
        """Non-symlink files are deleted and bytes counted."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            f = os.path.join(chat_dir, "photo.jpg")
            with open(f, "wb") as fh:
                fh.write(b"x" * 500)

            records = [{"file_path": f}]
            files, symlinks, freed, errors = _purge_chat_media_sync(records, media_path, 12345)

            assert files == 1
            assert symlinks == 0
            assert freed == 500
            assert errors == 0
            assert not os.path.exists(f)
            # Empty dir should be cleaned
            assert not os.path.exists(chat_dir)

    def test_removes_symlinks(self):
        """Symlinks are removed without touching the target."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared", "ab")
            os.makedirs(shared)
            blob = os.path.join(shared, "blob.jpg")
            with open(blob, "wb") as fh:
                fh.write(b"x" * 100)

            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "blob.jpg")
            try:
                os.symlink(blob, link)
            except OSError:
                pytest.skip("symlinks not supported")

            records = [{"file_path": link}]
            files, symlinks, freed, errors = _purge_chat_media_sync(records, media_path, 12345)

            assert files == 0
            assert symlinks == 1
            assert freed == 0
            assert errors == 0
            assert not os.path.lexists(link)
            # Blob should still exist
            assert os.path.exists(blob)

    def test_skips_missing_files(self):
        """Records with missing file_path are skipped."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            records = [
                {"file_path": None},
                {"file_path": "/nonexistent/file.jpg"},
                {},
            ]
            files, symlinks, freed, errors = _purge_chat_media_sync(records, td, 12345)
            assert files == 0
            assert symlinks == 0

    def test_keeps_nonempty_chat_dir(self):
        """Chat directory is NOT removed if other files remain."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            target = os.path.join(chat_dir, "delete_me.jpg")
            keeper = os.path.join(chat_dir, "keep_me.jpg")
            with open(target, "wb") as fh:
                fh.write(b"x")
            with open(keeper, "wb") as fh:
                fh.write(b"y")

            records = [{"file_path": target}]
            _purge_chat_media_sync(records, media_path, 12345)

            assert os.path.exists(chat_dir)
            assert os.path.exists(keeper)


class TestPurgeSkippedChatMedia:
    """Integration tests for purge_skipped_chat_media."""

    def _make_mock_db(self, chat_media: dict[int, list[dict]]) -> MagicMock:
        """Create a mock DB with media records keyed by chat_id."""
        db = MagicMock()

        async def _get_media(chat_id):
            return chat_media.get(chat_id, [])

        async def _delete_media(chat_id):
            return len(chat_media.get(chat_id, []))

        async def _iter_paths(batch_size=5000):
            # Return all file_paths across all chats
            all_paths = []
            for records in chat_media.values():
                for r in records:
                    if r.get("file_path"):
                        all_paths.append(r["file_path"])
            if all_paths:
                yield all_paths

        db.get_media_for_chat = _get_media
        db.delete_media_for_chat = _delete_media
        db.iter_all_media_file_paths = _iter_paths
        return db

    async def test_dry_run_reports_only(self):
        """Dry-run reports what would be deleted but doesn't touch anything."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            f = os.path.join(chat_dir, "photo.jpg")
            with open(f, "wb") as fh:
                fh.write(b"x" * 500)

            db = self._make_mock_db({12345: [{"file_path": f}]})
            result = await purge_skipped_chat_media(media_path, db, {12345}, delete=False)

            assert result["chats_processed"] == 1
            assert result["media_records"] == 1
            assert result["files_removed"] == 1
            assert result["db_records_deleted"] == 0
            # File should still exist
            assert os.path.exists(f)

    async def test_delete_removes_files_and_db_records(self):
        """Delete mode removes files and DB records."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            f = os.path.join(chat_dir, "photo.jpg")
            with open(f, "wb") as fh:
                fh.write(b"x" * 500)

            db = self._make_mock_db({12345: [{"file_path": f}]})
            result = await purge_skipped_chat_media(media_path, db, {12345}, delete=True)

            assert result["chats_processed"] == 1
            assert result["files_removed"] == 1
            assert result["freed_bytes"] == 500
            assert result["db_records_deleted"] == 1
            assert not os.path.exists(f)

    async def test_skips_chats_with_no_media(self):
        """Chats with no media records are skipped."""
        db = self._make_mock_db({})
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            result = await purge_skipped_chat_media(td, db, {12345, 67890}, delete=True)

            assert result["chats_processed"] == 0
            assert result["media_records"] == 0

    async def test_multiple_chats(self):
        """Multiple chats are processed."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            chat_media = {}
            for chat_id in (111, 222):
                chat_dir = os.path.join(media_path, str(chat_id))
                os.makedirs(chat_dir)
                f = os.path.join(chat_dir, "file.jpg")
                with open(f, "wb") as fh:
                    fh.write(b"x" * 100)
                chat_media[chat_id] = [{"file_path": f}]

            db = self._make_mock_db(chat_media)
            result = await purge_skipped_chat_media(media_path, db, {111, 222}, delete=True)

            assert result["chats_processed"] == 2
            assert result["files_removed"] == 2
            assert result["freed_bytes"] == 200

    async def test_purge_then_orphan_scan(self):
        """Purge + orphan scan together catches newly-orphaned blobs."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared", "ab")
            os.makedirs(shared)
            blob = os.path.join(shared, "blob.jpg")
            with open(blob, "wb") as fh:
                fh.write(b"x" * 1000)

            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "blob.jpg")
            try:
                os.symlink(blob, link)
            except OSError:
                pytest.skip("symlinks not supported")

            # DB references the symlink
            chat_media = {12345: [{"file_path": link}]}
            db = self._make_mock_db(chat_media)

            # Phase 1: purge removes symlink + DB records
            purge_result = await purge_skipped_chat_media(media_path, db, {12345}, delete=True)
            assert purge_result["symlinks_removed"] == 1
            assert "purged_realpaths" in purge_result
            assert not os.path.lexists(link)

            # After purge, DB has no references → update mock
            db_after = self._make_mock_db({})

            # Phase 2: orphan scan finds the now-unreferenced blob
            orphan_result = await clean_orphan_media(media_path, db_after, delete=True)
            assert orphan_result["orphan_blobs"] == 1
            assert orphan_result["deleted_blobs"] == 1
            assert orphan_result["freed_bytes"] == 1000
            assert not os.path.exists(blob)

    async def test_purged_realpaths_returned_in_dry_run(self):
        """Dry-run purge returns purged_realpaths for orphan scan simulation."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared", "ab")
            os.makedirs(shared)
            blob = os.path.join(shared, "blob.jpg")
            with open(blob, "wb") as fh:
                fh.write(b"x" * 100)

            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "blob.jpg")
            try:
                os.symlink(blob, link)
            except OSError:
                pytest.skip("symlinks not supported")

            db = self._make_mock_db({12345: [{"file_path": link}]})
            result = await purge_skipped_chat_media(media_path, db, {12345}, delete=False)

            assert "purged_realpaths" in result
            assert os.path.realpath(blob) in result["purged_realpaths"]

    async def test_dryrun_orphan_scan_with_exclude(self):
        """Dry-run orphan scan with exclude_realpaths shows accurate counts."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared", "ab")
            os.makedirs(shared)
            blob = os.path.join(shared, "blob.jpg")
            with open(blob, "wb") as fh:
                fh.write(b"x" * 500)

            chat_dir = os.path.join(media_path, "12345")
            os.makedirs(chat_dir)
            link = os.path.join(chat_dir, "blob.jpg")
            try:
                os.symlink(blob, link)
            except OSError:
                pytest.skip("symlinks not supported")

            # DB still references the symlink (dry-run didn't delete)
            db = self._make_mock_db({12345: [{"file_path": link}]})

            # Without exclude: no orphans
            result_no_exclude = await clean_orphan_media(media_path, db)
            assert result_no_exclude["orphan_blobs"] == 0

            # With exclude: blob is orphaned
            exclude = {os.path.realpath(blob)}
            result_with_exclude = await clean_orphan_media(media_path, db, exclude_realpaths=exclude)
            assert result_with_exclude["orphan_blobs"] == 1
            assert result_with_exclude["orphan_bytes"] == 500
            # Dry-run: nothing deleted
            assert result_with_exclude["deleted_blobs"] == 0
            # Blob still exists
            assert os.path.exists(blob)

    async def test_exclude_realpaths_shared_blob_safety(self):
        """Shared blob is NOT orphaned if another chat still references it."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            media_path = td
            shared = os.path.join(media_path, "_shared", "ab")
            os.makedirs(shared)
            blob = os.path.join(shared, "shared_blob.jpg")
            with open(blob, "wb") as fh:
                fh.write(b"x" * 500)

            # Two chats reference the same blob
            for chat_id in (12345, 67890):
                chat_dir = os.path.join(media_path, str(chat_id))
                os.makedirs(chat_dir)
                link = os.path.join(chat_dir, "shared_blob.jpg")
                try:
                    os.symlink(blob, link)
                except OSError:
                    pytest.skip("symlinks not supported")

            link_skipped = os.path.join(media_path, "12345", "shared_blob.jpg")
            link_kept = os.path.join(media_path, "67890", "shared_blob.jpg")

            # DB has both chats' references
            db = self._make_mock_db(
                {
                    12345: [{"file_path": link_skipped}],
                    67890: [{"file_path": link_kept}],
                }
            )

            # Only purging chat 12345 — chat 67890 still references the blob
            purge_result = await purge_skipped_chat_media(media_path, db, {12345}, delete=False)
            exclude = purge_result["purged_realpaths"]

            # Orphan scan with exclude: blob should NOT be orphaned
            # because chat 67890 still references it
            result = await clean_orphan_media(media_path, db, exclude_realpaths=exclude)
            assert result["orphan_blobs"] == 0
