"""Streaming downloader for direct-to-memory chunk loading."""

import io
import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional
from urllib import parse

from litdata.streaming.streaming_buffer import StreamingChunkBuffer

logger = logging.getLogger("litdata.streaming.streaming_downloader")


class StreamingDownloader(ABC):
    """Base class for streaming downloaders that load chunks directly into memory."""

    def __init__(self, storage_options: Optional[Dict] = None):
        self.storage_options = storage_options or {}

    @abstractmethod
    def stream_chunk(
        self,
        remote_filepath: str,
        buffer: StreamingChunkBuffer,
        chunk_index: int,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> bool:
        """Stream a chunk directly to memory buffer.

        Args:
            remote_filepath: Remote path to the chunk file
            buffer: Streaming buffer to store chunk data
            chunk_index: Index of the chunk being downloaded
            progress_callback: Optional callback for progress updates (chunk_index, progress)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def supports_url(self, url: str) -> bool:
        """Check if this downloader supports the given URL scheme."""
        pass


class S3StreamingDownloader(StreamingDownloader):
    """Streaming downloader for S3-compatible storage."""

    def __init__(self, storage_options: Optional[Dict] = None):
        super().__init__(storage_options)
        # Initialize S3 client with streaming support
        from litdata.streaming.client import S3Client

        self._client = S3Client(storage_options=self.storage_options)

    def supports_url(self, url: str) -> bool:
        """Check if this downloader supports S3 URLs."""
        return url.startswith("s3://")

    def stream_chunk(
        self,
        remote_filepath: str,
        buffer: StreamingChunkBuffer,
        chunk_index: int,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> bool:
        """Stream S3 chunk directly to memory buffer."""
        try:
            obj = parse.urlparse(remote_filepath)
            if obj.scheme != "s3":
                return False

            bucket_name = obj.netloc
            key = obj.path.lstrip("/")

            # Use BytesIO for streaming download
            file_obj = io.BytesIO()
            try:
                self._client.client.download_fileobj(bucket_name, key, file_obj)
            except Exception as e:
                logger.error(f"Failed to download S3 object {remote_filepath}: {e}")
                return False

            # Get the downloaded data
            file_obj.seek(0)
            chunk_data = file_obj.getvalue()

            if not chunk_data:
                logger.warning(f"Downloaded empty data for S3 chunk {chunk_index}")
                return False

            # Store in buffer
            success = buffer.store_chunk_data(chunk_index, chunk_data, is_complete=True)
            if not success:
                logger.warning(f"Failed to store chunk data for S3 chunk {chunk_index}")
                return False

            logger.info(f"Successfully streamed S3 chunk {chunk_index} ({len(chunk_data)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to stream S3 chunk {remote_filepath}: {e}")
            return False


class LocalStreamingDownloader(StreamingDownloader):
    """Streaming downloader for local files."""

    def supports_url(self, url: str) -> bool:
        """Check if this downloader supports local URLs."""
        return url.startswith("local:") or url.startswith("file://") or "://" not in url

    def stream_chunk(
        self,
        remote_filepath: str,
        buffer: StreamingChunkBuffer,
        chunk_index: int,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> bool:
        """Stream local file directly to memory buffer."""
        try:
            # Handle different local URL formats
            if remote_filepath.startswith("local:"):
                file_path = remote_filepath[6:]  # Remove "local:" prefix
            elif remote_filepath.startswith("file://"):
                file_path = remote_filepath[7:]  # Remove "file://" prefix
            else:
                file_path = remote_filepath

            import os

            if not os.path.exists(file_path):
                logger.error(f"Local file does not exist: {file_path}")
                return False

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.warning(f"Local file is empty: {file_path}")
                return False

            # Stream read in chunks
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            bytes_read = 0

            with open(file_path, "rb") as f:
                while bytes_read < file_size:
                    chunk_data = f.read(chunk_size)
                    if not chunk_data:
                        break

                    # Store in buffer
                    if bytes_read == 0:
                        success = buffer.store_chunk_data(chunk_index, chunk_data, is_complete=False)
                    else:
                        success = buffer.append_chunk_data(chunk_index, chunk_data)

                    if not success:
                        logger.warning(f"Failed to store chunk data for local chunk {chunk_index}")
                        return False

                    bytes_read += len(chunk_data)

                    # Update progress
                    progress = bytes_read / file_size
                    if progress_callback:
                        progress_callback(chunk_index, progress)

            logger.debug(f"Successfully streamed local chunk {chunk_index} ({bytes_read} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to stream local chunk {remote_filepath}: {e}")
            return False


class StreamingDownloaderManager:
    """Manages multiple streaming downloaders and routes requests."""

    def __init__(self, storage_options: Optional[Dict] = None):
        self.storage_options = storage_options or {}
        self._downloaders: List[StreamingDownloader] = []
        self._lock = threading.Lock()

        # Register default downloaders
        self._register_default_downloaders()

    def _register_default_downloaders(self) -> None:
        """Register default streaming downloaders."""
        try:
            self._downloaders.append(S3StreamingDownloader(self.storage_options))
        except Exception:
            logger.debug("S3 streaming downloader not available")
        # Always add local downloader
        self._downloaders.append(LocalStreamingDownloader(self.storage_options))

    def register_downloader(self, downloader: StreamingDownloader) -> None:
        """Register a custom streaming downloader."""
        with self._lock:
            self._downloaders.append(downloader)

    def get_downloader(self, url: str) -> Optional[StreamingDownloader]:
        """Get the appropriate downloader for a URL."""
        with self._lock:
            for downloader in self._downloaders:
                if downloader.supports_url(url):
                    return downloader
        return None

    def stream_chunk(
        self,
        remote_filepath: str,
        buffer: StreamingChunkBuffer,
        chunk_index: int,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> bool:
        """Stream a chunk using the appropriate downloader."""
        downloader = self.get_downloader(remote_filepath)
        if not downloader:
            logger.error(f"No suitable streaming downloader found for: {remote_filepath}")
            return False

        return downloader.stream_chunk(remote_filepath, buffer, chunk_index, progress_callback)
