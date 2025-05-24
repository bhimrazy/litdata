"""Streaming downloader for direct-to-memory chunk loading."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional
from urllib import parse

from litdata.constants import _AZURE_STORAGE_AVAILABLE, _GOOGLE_STORAGE_AVAILABLE, _HF_HUB_AVAILABLE
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
            key = obj.path.lstrip("/")            # Get object metadata to determine size
            try:
                response = self._client.client.head_object(Bucket=bucket_name, Key=key)
                total_size = response["ContentLength"]
            except Exception as e:
                logger.error(f"Failed to get object metadata for {remote_filepath}: {e}")
                return False

            # Stream download with range requests
            chunk_size = 64 * 1024  # 64KB chunks
            bytes_downloaded = 0

            while bytes_downloaded < total_size:
                # Calculate range for this request
                end_byte = min(bytes_downloaded + chunk_size - 1, total_size - 1)
                range_header = f"bytes={bytes_downloaded}-{end_byte}"

                try:
                    # Download this range
                    response = self._client.client.get_object(
                        Bucket=bucket_name, Key=key, Range=range_header
                    )

                    chunk_data = response["Body"].read()
                    if not chunk_data:
                        break

                    # Store in buffer (append mode for progressive loading)
                    if bytes_downloaded == 0:
                        success = buffer.store_chunk_data(chunk_index, chunk_data, is_complete=False)
                    else:
                        success = buffer.append_chunk_data(chunk_index, chunk_data)

                    if not success:
                        logger.warning(f"Failed to store chunk data for S3 chunk {chunk_index}")
                        return False

                    bytes_downloaded += len(chunk_data)

                    # Update progress
                    progress = bytes_downloaded / total_size
                    if progress_callback:
                        progress_callback(chunk_index, progress)

                except Exception as e:
                    logger.error(f"Failed to download range {range_header} for {remote_filepath}: {e}")
                    return False

            logger.debug(f"Successfully streamed S3 chunk {chunk_index} ({bytes_downloaded} bytes)")
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
            chunk_size = 64 * 1024  # 64KB chunks
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


class GCPStreamingDownloader(StreamingDownloader):
    """Streaming downloader for Google Cloud Storage."""

    def __init__(self, storage_options: Optional[Dict] = None):
        super().__init__(storage_options)
        if not _GOOGLE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError("Google Cloud Storage support not available")

        from google.cloud import storage

        self._client = storage.Client()

    def supports_url(self, url: str) -> bool:
        """Check if this downloader supports GCS URLs."""
        return url.startswith("gs://")

    def stream_chunk(
        self,
        remote_filepath: str,
        buffer: StreamingChunkBuffer,
        chunk_index: int,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> bool:
        """Stream GCS chunk directly to memory buffer."""
        try:
            obj = parse.urlparse(remote_filepath)
            if obj.scheme != "gs":
                return False

            bucket_name = obj.netloc
            blob_name = obj.path.lstrip("/")

            bucket = self._client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Get blob size
            blob.reload()
            total_size = blob.size

            # Stream download with range requests
            chunk_size = 64 * 1024  # 64KB chunks
            bytes_downloaded = 0

            while bytes_downloaded < total_size:
                # Calculate range for this request
                end_byte = min(bytes_downloaded + chunk_size - 1, total_size - 1)

                try:
                    # Download this range
                    chunk_data = blob.download_as_bytes(start=bytes_downloaded, end=end_byte + 1)

                    if not chunk_data:
                        break

                    # Store in buffer
                    if bytes_downloaded == 0:
                        success = buffer.store_chunk_data(chunk_index, chunk_data, is_complete=False)
                    else:
                        success = buffer.append_chunk_data(chunk_index, chunk_data)

                    if not success:
                        logger.warning(f"Failed to store chunk data for GCS chunk {chunk_index}")
                        return False

                    bytes_downloaded += len(chunk_data)

                    # Update progress
                    progress = bytes_downloaded / total_size
                    if progress_callback:
                        progress_callback(chunk_index, progress)

                except Exception as e:
                    logger.error(f"Failed to download range {bytes_downloaded}-{end_byte} for {remote_filepath}: {e}")
                    return False

            logger.debug(f"Successfully streamed GCS chunk {chunk_index} ({bytes_downloaded} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to stream GCS chunk {remote_filepath}: {e}")
            return False


class AzureStreamingDownloader(StreamingDownloader):
    """Streaming downloader for Azure Blob Storage."""

    def __init__(self, storage_options: Optional[Dict] = None):
        super().__init__(storage_options)
        if not _AZURE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError("Azure Blob Storage support not available")

    def supports_url(self, url: str) -> bool:
        """Check if this downloader supports Azure URLs."""
        return url.startswith("azure://")

    def stream_chunk(
        self,
        remote_filepath: str,
        buffer: StreamingChunkBuffer,
        chunk_index: int,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> bool:
        """Stream Azure blob directly to memory buffer."""
        try:
            # TODO: Implement Azure streaming download
            # This would be similar to S3 but using Azure SDK
            logger.warning("Azure streaming download not yet implemented")
            return False

        except Exception as e:
            logger.error(f"Failed to stream Azure chunk {remote_filepath}: {e}")
            return False


class HFStreamingDownloader(StreamingDownloader):
    """Streaming downloader for Hugging Face Hub."""

    def __init__(self, storage_options: Optional[Dict] = None):
        super().__init__(storage_options)
        if not _HF_HUB_AVAILABLE:
            raise ModuleNotFoundError("Hugging Face Hub support not available")

    def supports_url(self, url: str) -> bool:
        """Check if this downloader supports HF URLs."""
        return url.startswith("hf://")

    def stream_chunk(
        self,
        remote_filepath: str,
        buffer: StreamingChunkBuffer,
        chunk_index: int,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> bool:
        """Stream HF dataset file directly to memory buffer."""
        try:
            # TODO: Implement HF streaming download
            # This would require HF Hub streaming APIs
            logger.warning("Hugging Face streaming download not yet implemented")
            return False

        except Exception as e:
            logger.error(f"Failed to stream HF chunk {remote_filepath}: {e}")
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

        try:
            self._downloaders.append(GCPStreamingDownloader(self.storage_options))
        except Exception:
            logger.debug("GCP streaming downloader not available")

        try:
            self._downloaders.append(AzureStreamingDownloader(self.storage_options))
        except Exception:
            logger.debug("Azure streaming downloader not available")

        try:
            self._downloaders.append(HFStreamingDownloader(self.storage_options))
        except Exception:
            logger.debug("HF streaming downloader not available")

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
