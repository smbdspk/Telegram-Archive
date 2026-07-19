"""Detect and remove orphan media blobs in the ``_shared/`` dedup store.

An *orphan* is a file inside ``_shared/`` (including shard buckets) that no
``Media`` database record references via ``file_path``.  Orphans accumulate
when:

* ``_cleanup_existing_media()`` removes symlinks + DB records for
  ``SKIP_MEDIA_CHAT_IDS`` chats but leaves the underlying blob.
* A download was interrupted after writing the blob but before committing
  the ``Media`` row.
* Manual DB edits or past bugs removed references.

Usage (via CLI)::

    # Dry-run — report only
    python -m src clean-media

    # Actually delete orphans
    python -m src clean-media --delete

    # Purge per-chat media for SKIP_MEDIA_CHAT_IDS chats, then clean orphans
    python -m src clean-media --purge-skipped --delete

    # Also clean up dangling symlinks in per-chat dirs
    python -m src clean-media --delete --include-dangling
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_SHARED_DIR_NAME = "_shared"


class _CleanupDB(Protocol):
    """Narrow DB surface the cleanup depends on."""

    def iter_all_media_file_paths(self, batch_size: int = ...) -> AsyncIterator[list[str]]: ...

    async def get_media_for_chat(self, chat_id: int) -> list[dict[str, Any]]: ...

    async def delete_media_for_chat(self, chat_id: int) -> int: ...


# ---------------------------------------------------------------------------
# Filesystem helpers (blocking — run via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _collect_shared_blobs(shared_dir: str) -> dict[str, int]:
    """Walk ``_shared/`` and return ``{realpath: size_bytes}`` for every file.

    Handles both sharded (``<hash[:2]>/<file>``) and flat (pre-sharding)
    layouts.  Skips subdirectories that aren't 2-char shard buckets at the
    first level (e.g. marker files from the repair pass).
    """
    blobs: dict[str, int] = {}
    if not os.path.isdir(shared_dir):
        return blobs

    with os.scandir(shared_dir) as it:
        for entry in it:
            if entry.is_file(follow_symlinks=False):
                # Flat (pre-sharding) blob
                try:
                    blobs[os.path.realpath(entry.path)] = entry.stat().st_size
                except OSError:
                    pass
            elif entry.is_dir(follow_symlinks=False) and len(entry.name) == 2:
                # Shard bucket
                try:
                    with os.scandir(entry.path) as bucket:
                        for child in bucket:
                            if child.is_file(follow_symlinks=False):
                                try:
                                    blobs[os.path.realpath(child.path)] = child.stat().st_size
                                except OSError:
                                    pass
                except OSError:
                    pass
    return blobs


def _collect_dangling_symlinks(media_path: str) -> list[str]:
    """Find symlinks in per-chat directories that point to missing targets.

    Skips ``_shared/`` and ``avatars/``.
    """
    dangling: list[str] = []
    if not os.path.isdir(media_path):
        return dangling

    try:
        entries = list(os.scandir(media_path))
    except OSError:
        return dangling

    for entry in entries:
        if entry.name in (_SHARED_DIR_NAME, "avatars"):
            continue
        if not entry.is_dir(follow_symlinks=False):
            continue
        try:
            with os.scandir(entry.path) as chat_dir:
                for child in chat_dir:
                    if child.is_symlink() and not os.path.exists(child.path):
                        dangling.append(child.path)
        except OSError:
            pass

    return dangling


def _delete_orphans_sync(
    orphan_paths: list[str],
    shared_dir: str,
) -> tuple[int, int, int]:
    """Remove orphan blobs and clean up empty shard dirs.

    Returns ``(deleted_count, freed_bytes, errors)``.
    """
    deleted = 0
    freed = 0
    errors = 0

    for path in orphan_paths:
        try:
            size = os.path.getsize(path)
            os.remove(path)
            deleted += 1
            freed += size
        except FileNotFoundError:
            # TOCTOU — another process already removed it
            pass
        except OSError:
            errors += 1

    # Clean up empty shard bucket directories
    if os.path.isdir(shared_dir):
        try:
            with os.scandir(shared_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False) and len(entry.name) == 2:
                        try:
                            if not os.listdir(entry.path):
                                os.rmdir(entry.path)
                        except OSError:
                            pass
        except OSError:
            pass

    return deleted, freed, errors


def _delete_dangling_sync(dangling_paths: list[str]) -> tuple[int, int]:
    """Remove dangling symlinks. Returns ``(deleted, errors)``."""
    deleted = 0
    errors = 0
    for path in dangling_paths:
        try:
            os.unlink(path)
            deleted += 1
        except FileNotFoundError:
            pass
        except OSError:
            errors += 1
    return deleted, errors


def _purge_chat_media_sync(
    media_records: list[dict[str, Any]],
    media_path: str,
    chat_id: int,
) -> tuple[int, int, int, int]:
    """Remove per-chat symlinks/files for a single chat.

    Returns ``(deleted_files, deleted_symlinks, freed_bytes, errors)``.
    Only removes files in the per-chat directory — never touches
    ``_shared/`` blobs directly (the orphan scan handles those).
    """
    deleted_files = 0
    deleted_symlinks = 0
    freed_bytes = 0
    errors = 0

    for record in media_records:
        file_path = record.get("file_path")
        if not file_path or not os.path.exists(file_path):
            continue
        try:
            if os.path.islink(file_path):
                os.unlink(file_path)
                deleted_symlinks += 1
            else:
                freed_bytes += os.path.getsize(file_path)
                os.remove(file_path)
                deleted_files += 1
        except FileNotFoundError:
            pass
        except OSError:
            errors += 1

    # Clean up empty chat media directory
    chat_media_dir = os.path.join(media_path, str(chat_id))
    if os.path.isdir(chat_media_dir):
        try:
            if not os.listdir(chat_media_dir):
                os.rmdir(chat_media_dir)
        except OSError:
            pass

    return deleted_files, deleted_symlinks, freed_bytes, errors


async def purge_skipped_chat_media(
    media_path: str,
    db: _CleanupDB,
    skip_chat_ids: set[int],
    *,
    delete: bool = False,
) -> dict:
    """Remove per-chat symlinks + DB records for ``SKIP_MEDIA_CHAT_IDS`` chats.

    This is the per-chat cleanup that ``_cleanup_existing_media()`` does inside
    the backup loop, but callable standalone — no Telegram API needed.

    After this runs, any blobs in ``_shared/`` that lost their last reference
    will be caught by the orphan scan (``clean_orphan_media``).

    Args:
        media_path: Root media directory.
        db: Database adapter.
        skip_chat_ids: Chat IDs to purge media for.
        delete: If True, actually delete. Otherwise report only.

    Returns:
        Summary dict with per-chat and aggregate counts.
    """
    total_records = 0
    total_files = 0
    total_symlinks = 0
    total_freed = 0
    total_errors = 0
    total_db_deleted = 0
    chats_processed = 0

    for chat_id in sorted(skip_chat_ids):
        media_records = await db.get_media_for_chat(chat_id)
        if not media_records:
            continue

        total_records += len(media_records)
        chats_processed += 1

        if not delete:
            # Dry-run: count what would be deleted
            for record in media_records:
                file_path = record.get("file_path")
                if file_path and os.path.exists(file_path):
                    if os.path.islink(file_path):
                        total_symlinks += 1
                    else:
                        total_files += 1
                        try:
                            total_freed += os.path.getsize(file_path)
                        except OSError:
                            pass
            continue

        # Delete mode: remove files, then DB records
        files, symlinks, freed, errs = await asyncio.to_thread(
            _purge_chat_media_sync, media_records, media_path, chat_id
        )
        total_files += files
        total_symlinks += symlinks
        total_freed += freed
        total_errors += errs

        db_deleted = await db.delete_media_for_chat(chat_id)
        total_db_deleted += db_deleted

    if total_records:
        logger.info(
            "Purge skipped chats: %d chats, %d records, %d files, %d symlinks",
            chats_processed,
            total_records,
            total_files,
            total_symlinks,
        )

    return {
        "chats_processed": chats_processed,
        "media_records": total_records,
        "files_removed": total_files,
        "symlinks_removed": total_symlinks,
        "freed_bytes": total_freed,
        "db_records_deleted": total_db_deleted,
        "errors": total_errors,
    }


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------


async def clean_orphan_media(
    media_path: str,
    db: _CleanupDB,
    *,
    delete: bool = False,
    include_dangling: bool = False,
) -> dict:
    """Detect (and optionally remove) orphan media blobs.

    Args:
        media_path: Root media directory (``config.media_path``).
        db: Database adapter implementing ``iter_all_media_file_paths``.
        delete: If True, actually remove orphans. Otherwise report only.
        include_dangling: If True, also detect/remove dangling symlinks
            in per-chat directories.

    Returns:
        Summary dict with counts and sizes.
    """
    shared_dir = os.path.join(media_path, _SHARED_DIR_NAME)

    # 1. Collect all blobs on disk
    blobs = await asyncio.to_thread(_collect_shared_blobs, shared_dir)
    logger.info("Found %d blobs in _shared/", len(blobs))

    if not blobs:
        result = {
            "total_blobs": 0,
            "referenced_blobs": 0,
            "orphan_blobs": 0,
            "orphan_bytes": 0,
            "deleted_blobs": 0,
            "freed_bytes": 0,
            "errors": 0,
        }
        if include_dangling:
            result.update(dangling_symlinks=0, deleted_dangling=0, dangling_errors=0)
        return result

    # 2. Collect all referenced paths from DB and resolve to real paths
    referenced_realpaths: set[str] = set()
    async for batch in db.iter_all_media_file_paths():
        for file_path in batch:
            # Resolve symlinks to the underlying _shared/ blob
            try:
                real = os.path.realpath(file_path)
                referenced_realpaths.add(real)
            except OSError, ValueError:
                pass

    logger.info("DB references %d unique real paths", len(referenced_realpaths))

    # 3. Diff: orphans = on-disk blobs NOT in the referenced set
    orphan_paths = [p for p in blobs if p not in referenced_realpaths]
    orphan_bytes = sum(blobs[p] for p in orphan_paths)

    logger.info(
        "Orphan analysis: %d orphans (%.1f MB) out of %d total blobs",
        len(orphan_paths),
        orphan_bytes / (1024 * 1024),
        len(blobs),
    )

    # 4. Optionally detect dangling symlinks
    dangling_paths: list[str] = []
    if include_dangling:
        dangling_paths = await asyncio.to_thread(_collect_dangling_symlinks, media_path)
        if dangling_paths:
            logger.info("Found %d dangling symlinks in per-chat directories", len(dangling_paths))

    # 5. Delete if requested
    deleted_blobs = 0
    freed_bytes = 0
    blob_errors = 0
    deleted_dangling = 0
    dangling_errors = 0

    if delete and orphan_paths:
        deleted_blobs, freed_bytes, blob_errors = await asyncio.to_thread(
            _delete_orphans_sync, orphan_paths, shared_dir
        )
        logger.info(
            "Deleted %d orphan blobs (%.1f MB freed, %d errors)",
            deleted_blobs,
            freed_bytes / (1024 * 1024),
            blob_errors,
        )

    if delete and dangling_paths:
        deleted_dangling, dangling_errors = await asyncio.to_thread(_delete_dangling_sync, dangling_paths)
        if deleted_dangling:
            logger.info("Deleted %d dangling symlinks (%d errors)", deleted_dangling, dangling_errors)

    result = {
        "total_blobs": len(blobs),
        "referenced_blobs": len(blobs) - len(orphan_paths),
        "orphan_blobs": len(orphan_paths),
        "orphan_bytes": orphan_bytes,
        "deleted_blobs": deleted_blobs,
        "freed_bytes": freed_bytes,
        "errors": blob_errors,
    }
    if include_dangling:
        result.update(
            dangling_symlinks=len(dangling_paths),
            deleted_dangling=deleted_dangling,
            dangling_errors=dangling_errors,
        )
    return result
