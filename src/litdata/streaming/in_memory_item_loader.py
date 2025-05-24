"""In-memory item loader for streaming data access."""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from litdata.streaming.item_loader import BaseItemLoader, Interval
from litdata.streaming.streaming_buffer import StreamingChunkBuffer, StreamingConfig
from litdata.streaming.streaming_downloader import StreamingDownloaderManager
from litdata.utilities._pytree import tree_unflatten

logger = logging.getLogger("litdata.streaming.in_memory_item_loader")


class InMemoryItemLoader(BaseItemLoader):
    """In-memory item loader that provides fast access to streaming data.

    This loader keeps chunks in memory for faster access while supporting
    progressive loading and memory management.
    """

    def __init__(self, streaming_config: Optional[StreamingConfig] = None):
        """Initialize the in-memory item loader.

        Args:
            streaming_config: Configuration for streaming buffer behavior
        """
        super().__init__()
        self.streaming_config = streaming_config or StreamingConfig()
        self.chunk_buffer = StreamingChunkBuffer(self.streaming_config)
        self.downloader_manager = StreamingDownloaderManager()
        self._lock = threading.RLock()
        self._chunk_downloaders: Dict[int, threading.Thread] = {}
        self._download_progress: Dict[int, float] = {}
        self.region_of_interest = None  # Add missing attribute

    def generate_intervals(self, chunks: Optional[List[Dict]] = None) -> List[Interval]:
        """Generate intervals for chunk loading.

        Args:
            chunks: List of chunk information dicts. If None, uses self._chunks
        """
        chunks_to_use = chunks if chunks is not None else self._chunks
        intervals = []
        begin = 0
        end = 0

        # Create a custom class that provides both the expected interface and Interval compatibility
        class CustomInterval:
            def __init__(self, chunk_index: int, chunk_start: int, roi_start: int, roi_end: int, chunk_end: int):
                # Interval namedtuple compatibility
                self._tuple = Interval(chunk_start, roi_start, roi_end, chunk_end)
                # Test-expected attributes
                self.chunk_index = chunk_index
                self.start = roi_start
                self.stop = roi_end
                
                # Provide access to Interval namedtuple fields
                self.chunk_start = chunk_start
                self.roi_start_idx = roi_start
                self.roi_end_idx = roi_end
                self.chunk_end = chunk_end
            
            # Make it behave like a tuple for compatibility
            def __getitem__(self, index):
                return self._tuple[index]
            
            def __len__(self):
                return len(self._tuple)
            
            def __iter__(self):
                return iter(self._tuple)

        for idx, curr_chunk in enumerate(chunks_to_use):
            chunk_size = curr_chunk.get("dim", curr_chunk.get("chunk_size", 0))
            
            # Handle case where chunk_size might be None
            if chunk_size is None:
                # Try alternative fields that might contain the chunk size
                chunk_size = curr_chunk.get("samples", curr_chunk.get("length", 0))
                if chunk_size is None:
                    chunk_size = 0
                    
            end += chunk_size
            start_idx, end_idx = begin, end

            if self.region_of_interest is not None and idx < len(self.region_of_interest):
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]

            # Create custom interval that satisfies both test expectations and system compatibility
            intervals.append(CustomInterval(idx, begin, start_idx, end_idx, end))
            begin += chunk_size

        return intervals

    def pre_load_chunk(self, chunk_filepath: str, chunk_index: int) -> bool:
        """Pre-load chunk into memory buffer.

        Args:
            chunk_filepath: Path to the chunk file
            chunk_index: Index of the chunk

        Returns:
            True if chunk was successfully loaded, False otherwise
        """
        with self._lock:
            # Check if chunk is already available
            if self.chunk_buffer.is_chunk_available(chunk_index):
                return True

            # Try to load chunk using downloader manager
            success = self.downloader_manager.stream_chunk(chunk_filepath, self.chunk_buffer, chunk_index)
            return success

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        filesize_bytes: int,
    ) -> Any:
        """Load a specific item from a chunk.

        Args:
            index: Item index within the chunk
            chunk_index: Index of the chunk
            chunk_filepath: Path to the chunk file
            begin: Beginning offset
            filesize_bytes: Size of the file

        Returns:
            The loaded item
        """
        # Try to get item from memory buffer first
        item_data = self._get_item_from_buffer(chunk_index, index)
        if item_data is not None:
            return self._deserialize_item(item_data, index)

        # If not in buffer, ensure chunk is being downloaded
        self._ensure_chunk_download(chunk_index, chunk_filepath)

        # Wait for item to become available or fallback to direct loading
        max_wait_time = 5.0  # 5 seconds timeout
        wait_interval = 0.1
        elapsed_time = 0.0

        while elapsed_time < max_wait_time:
            item_data = self._get_item_from_buffer(chunk_index, index)
            if item_data is not None:
                return self._deserialize_item(item_data, index)

            # Check if we have enough partial data
            available_items = self.chunk_buffer.get_available_items(chunk_index)
            if index in available_items:
                item_data = self._get_item_from_buffer(chunk_index, index)
                if item_data is not None:
                    return self._deserialize_item(item_data, index)

            threading.Event().wait(wait_interval)
            elapsed_time += wait_interval

        # Fallback: direct file access if streaming fails
        logger.warning(f"Streaming failed for chunk {chunk_index}, item {index}. Falling back to direct file access.")
        return self._load_item_directly(index, chunk_index, chunk_filepath, begin, filesize_bytes)

    def delete(self) -> None:
        """Delete and cleanup all loaded data."""
        with self._lock:
            # Clear all chunks from buffer
            self.chunk_buffer.clear_all_chunks()

            # Cancel any ongoing downloads
            self._chunk_downloaders.clear()

            # Remove download progress tracking
            self._download_progress.clear()

            logger.info("In-memory item loader cleaned up")

    def encode_data(self, data: List[bytes], sizes: List[int], flattened: List[Any]) -> Any:
        """Encode data for storage."""
        # Use the same encoding logic as PyTreeLoader
        return tree_unflatten(flattened)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        buffer_stats = self.chunk_buffer.get_memory_usage()
        return {
            "total_memory_usage": buffer_stats["current_usage"],
            "max_memory_size": buffer_stats["max_size"],
            "num_chunks_cached": buffer_stats["num_chunks"],
            "memory_utilization": buffer_stats["usage_ratio"],
            "chunks": buffer_stats["chunks"],
        }

    def get_chunk_progress(self, chunk_index: int) -> float:
        """Get download progress for a chunk (0.0 to 1.0)."""
        return self._download_progress.get(chunk_index, 0.0)

    def is_item_available(self, chunk_index: int, item_index: int) -> bool:
        """Check if a specific item is available in memory."""
        available_items = self.chunk_buffer.get_available_items(chunk_index)
        return item_index in available_items

    def prefetch_items(self, chunk_index: int, start_item: int, num_items: int) -> None:
        """Prefetch a range of items from a chunk."""
        chunk_info = self._chunks[chunk_index] if chunk_index < len(self._chunks) else None
        if not chunk_info:
            return

        chunk_filepath = chunk_info.get("filename", "")
        if chunk_filepath:
            self.pre_load_chunk(chunk_filepath, chunk_index)

    def _download_chunk_background(self, chunk_index: int, chunk_filepath: str) -> None:
        """Download chunk in background with progress tracking using streaming downloader."""
        try:
            # Get the remote URL from chunk configuration
            remote_filepath = self._get_remote_filepath(chunk_index, chunk_filepath)

            if not remote_filepath:
                logger.warning(f"Could not determine remote filepath for chunk {chunk_index}")
                return

            # Use streaming downloader to download directly to memory
            def progress_callback(idx: int, progress: float) -> None:
                self._download_progress[idx] = progress

            success = self.downloader_manager.stream_chunk(
                remote_filepath, self.chunk_buffer, chunk_index
            )

            if success:
                logger.debug(f"Successfully streamed chunk {chunk_index} from {remote_filepath}")
                self._download_progress[chunk_index] = 1.0
            else:
                logger.warning(f"Failed to stream chunk {chunk_index}, falling back to disk-based loading")
                # Fallback to original disk-based approach
                self._download_chunk_from_disk(chunk_index, chunk_filepath)

        except Exception as e:
            logger.error(f"Error streaming chunk {chunk_index}: {e}")
            # Fallback to original disk-based approach
            self._download_chunk_from_disk(chunk_index, chunk_filepath)
        finally:
            # Clean up
            with self._lock:
                self._chunk_downloaders.pop(chunk_index, None)

    def _get_remote_filepath(self, chunk_index: int, chunk_filepath: str) -> Optional[str]:
        """Get remote filepath from chunk configuration."""
        try:
            # Get remote directory from config
            if hasattr(self.streaming_config, "_remote_dir") and self.streaming_config._remote_dir:
                remote_dir = self.streaming_config._remote_dir
                chunk_filename = os.path.basename(chunk_filepath)
                return os.path.join(remote_dir, chunk_filename)

            # If no remote directory, check if chunk_filepath is already a remote URL
            if any(
                chunk_filepath.startswith(scheme)
                for scheme in ["s3://", "gs://", "azure://", "hf://", "http://", "https://"]
            ):
                return chunk_filepath

            # Default to local file
            return f"local:{chunk_filepath}"

        except Exception as e:
            logger.debug(f"Error determining remote filepath for chunk {chunk_index}: {e}")
            return None

    def _download_chunk_from_disk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Fallback method: download chunk from disk (original approach)."""
        try:
            if not os.path.exists(chunk_filepath):
                logger.warning(f"Chunk file does not exist: {chunk_filepath}")
                return

            file_size = os.path.getsize(chunk_filepath)
            if file_size == 0:
                logger.warning(f"Chunk file is empty: {chunk_filepath}")
                return

            # Open file and read in chunks for progressive loading
            with open(chunk_filepath, "rb") as f:
                buffer_size = 64 * 1024  # 64KB chunks
                data_buffer = b""
                bytes_read = 0

                while True:
                    chunk_data = f.read(buffer_size)
                    if not chunk_data:
                        break

                    data_buffer += chunk_data
                    bytes_read += len(chunk_data)

                    # Update progress
                    progress = bytes_read / file_size
                    self._download_progress[chunk_index] = progress

                    # Store partial data in buffer
                    is_complete = bytes_read >= file_size
                    if len(data_buffer) >= buffer_size or is_complete:
                        success = self.chunk_buffer.append_chunk_data(chunk_index, data_buffer)
                        if not success:
                            logger.warning(f"Failed to store chunk data for chunk {chunk_index}")
                            break
                        data_buffer = b""

                    if is_complete:
                        break

                logger.debug(f"Completed downloading chunk {chunk_index} from disk ({bytes_read} bytes)")

        except Exception as e:
            logger.error(f"Error downloading chunk from disk {chunk_index}: {e}")

    def _ensure_chunk_download(self, chunk_index: int, chunk_filepath: str) -> None:
        """Ensure chunk download is started."""
        with self._lock:
            if chunk_index not in self._chunk_downloaders and not self.chunk_buffer.is_chunk_available(chunk_index):
                self.pre_load_chunk(chunk_filepath, chunk_index)

    def _get_item_from_buffer(self, chunk_index: int, item_index: int) -> Optional[bytes]:
        """Get item data from memory buffer."""
        if not self.chunk_buffer.is_chunk_available(chunk_index):
            return None

        # Get chunk metadata to find item offset
        metadata = self.chunk_buffer._metadata.get(chunk_index)
        if not metadata or not metadata.item_offsets:
            return None

        if item_index >= len(metadata.item_offsets) - 1:
            return None

        # Calculate item boundaries
        item_start = metadata.item_offsets[item_index]
        item_end = metadata.item_offsets[item_index + 1]
        item_size = item_end - item_start

        # Get item data
        return self.chunk_buffer.get_partial_chunk_data(chunk_index, item_start, item_size)

    def _deserialize_item(self, item_data: bytes, item_index: int) -> Any:
        """Deserialize item data using the same logic as PyTreeLoader."""
        try:
            # Use the same deserialization logic as PyTreeLoader
            # Format: [size_header][concatenated_data]
            # where size_header contains the byte sizes of each object encoded as uint32
            
            idx = self._shift_idx  # Skip the size header
            if len(item_data) < idx:
                raise ValueError("Item data too short for size header")
                
            # Read the sizes from the header
            sizes = np.frombuffer(item_data[:idx], np.uint32)
            
            # Deserialize each data segment
            data = []
            for size, data_format in zip(sizes, self._data_format):
                if idx + size > len(item_data):
                    raise ValueError(f"Item data truncated at index {idx}, expected {size} more bytes")
                    
                serializer = self._serializers[data_format]
                data_bytes = item_data[idx : idx + size]
                data.append(serializer.deserialize(data_bytes))
                idx += size
                
            # Reconstruct the original object using PyTree
            return tree_unflatten(data, self._config["data_spec"])

        except Exception as e:
            logger.error(f"Failed to deserialize item {item_index}: {e}")
            # Log first few bytes for debugging
            data_preview = item_data[:min(50, len(item_data))]
            logger.error(f"Item data preview (first 50 bytes): {data_preview}")
            return None

    def _load_item_directly(
        self, index: int, chunk_index: int, chunk_filepath: str, begin: int, filesize_bytes: int
    ) -> Any:
        """Fallback: load item directly from file."""
        try:
            with open(chunk_filepath, "rb") as f:
                # Read chunk header to get item offsets
                f.seek(0)
                header_data = f.read(4)
                if len(header_data) < 4:
                    raise ValueError("Invalid chunk header")

                num_items = np.frombuffer(header_data, np.uint32)[0]

                # Read offset array
                offset_array_size = (num_items + 1) * 4
                offset_data = f.read(offset_array_size)
                offsets = np.frombuffer(offset_data, np.uint32)

                if index >= len(offsets) - 1:
                    raise ValueError(f"Item index {index} out of range")

                # Read item data
                item_start = offsets[index]
                item_end = offsets[index + 1]

                f.seek(item_start)
                item_data = f.read(item_end - item_start)

                return self._deserialize_item(item_data, index)

        except Exception as e:
            logger.error(f"Failed to load item {index} directly from {chunk_filepath}: {e}")
            return None
