"""In-memory item loader for streaming data access."""

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from litdata.streaming.item_loader import BaseItemLoader, Interval
from litdata.streaming.streaming_buffer import MemoryManager, StreamingConfig
from litdata.utilities._pytree import tree_unflatten

logger = logging.getLogger("litdata.streaming.in_memory_item_loader")


class InMemoryItemLoader(BaseItemLoader):
    """In-memory item loader that provides fast access to streaming data.

    This loader keeps chunks in memory for faster access while supporting
    progressive loading and memory management.
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        """Initialize the in-memory item loader.

        Args:
            config: Configuration for streaming buffer behavior
        """
        super().__init__()
        self.config = config or StreamingConfig()
        self.memory_manager = MemoryManager(self.config)
        self._lock = threading.RLock()
        self._chunk_downloaders: Dict[int, threading.Thread] = {}
        self._download_progress: Dict[int, float] = {}

    def generate_intervals(self) -> List[Interval]:
        """Generate intervals for chunk loading."""
        intervals = []
        begin = 0
        end = 0

        for idx, curr_chunk in enumerate(self._chunks):
            end += curr_chunk["chunk_size"]
            start_idx, end_idx = begin, end

            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]

            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += curr_chunk["chunk_size"]

        return intervals

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Pre-load chunk into memory buffer."""
        with self._lock:
            # Check if chunk is already being downloaded or available
            if chunk_index in self._chunk_downloaders or self.memory_manager.buffer.is_chunk_available(chunk_index):
                return

            # Start background download
            download_thread = threading.Thread(
                target=self._download_chunk_background, args=(chunk_index, chunk_filepath), daemon=True
            )
            self._chunk_downloaders[chunk_index] = download_thread
            download_thread.start()

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
            available_items = self.memory_manager.buffer.get_available_items(chunk_index)
            if index in available_items:
                item_data = self._get_item_from_buffer(chunk_index, index)
                if item_data is not None:
                    return self._deserialize_item(item_data, index)

            threading.Event().wait(wait_interval)
            elapsed_time += wait_interval

        # Fallback: direct file access if streaming fails
        logger.warning(f"Streaming failed for chunk {chunk_index}, item {index}. Falling back to direct file access.")
        return self._load_item_directly(index, chunk_index, chunk_filepath, begin, filesize_bytes)

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete chunk from memory and disk."""
        with self._lock:
            # Remove from memory buffer
            self.memory_manager.buffer.remove_chunk(chunk_index)

            # Cancel any ongoing downloads
            if chunk_index in self._chunk_downloaders:
                # Note: We can't actually cancel the thread, but we can mark it as cancelled
                del self._chunk_downloaders[chunk_index]

            # Remove download progress tracking
            self._download_progress.pop(chunk_index, None)

        # Delete from disk if exists
        if os.path.exists(chunk_filepath):
            try:
                os.remove(chunk_filepath)
                logger.debug(f"Deleted chunk file: {chunk_filepath}")
            except OSError as e:
                logger.warning(f"Failed to delete chunk file {chunk_filepath}: {e}")

    def encode_data(self, data: List[bytes], sizes: List[int], flattened: List[Any]) -> Any:
        """Encode data for storage."""
        # Use the same encoding logic as PyTreeLoader
        return tree_unflatten(flattened)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return self.memory_manager.buffer.get_memory_usage()

    def get_chunk_progress(self, chunk_index: int) -> float:
        """Get download progress for a chunk (0.0 to 1.0)."""
        return self._download_progress.get(chunk_index, 0.0)

    def is_item_available(self, chunk_index: int, item_index: int) -> bool:
        """Check if a specific item is available in memory."""
        available_items = self.memory_manager.buffer.get_available_items(chunk_index)
        return item_index in available_items

    def prefetch_items(self, chunk_index: int, start_item: int, num_items: int) -> None:
        """Prefetch a range of items from a chunk."""
        chunk_info = self._chunks[chunk_index] if chunk_index < len(self._chunks) else None
        if not chunk_info:
            return

        chunk_filepath = chunk_info.get("filename", "")
        if chunk_filepath:
            self.pre_load_chunk(chunk_index, chunk_filepath)

    def _download_chunk_background(self, chunk_index: int, chunk_filepath: str) -> None:
        """Download chunk in background with progress tracking."""
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
                        success = self.memory_manager.buffer.append_chunk_data(chunk_index, data_buffer)
                        if not success:
                            logger.warning(f"Failed to store chunk data for chunk {chunk_index}")
                            break
                        data_buffer = b""

                    if is_complete:
                        break

                logger.debug(f"Completed downloading chunk {chunk_index} ({bytes_read} bytes)")

        except Exception as e:
            logger.error(f"Error downloading chunk {chunk_index}: {e}")
        finally:
            # Clean up
            with self._lock:
                self._chunk_downloaders.pop(chunk_index, None)

    def _ensure_chunk_download(self, chunk_index: int, chunk_filepath: str) -> None:
        """Ensure chunk download is started."""
        with self._lock:
            if chunk_index not in self._chunk_downloaders and not self.memory_manager.buffer.is_chunk_available(
                chunk_index
            ):
                self.pre_load_chunk(chunk_index, chunk_filepath)

    def _get_item_from_buffer(self, chunk_index: int, item_index: int) -> Optional[bytes]:
        """Get item data from memory buffer."""
        if not self.memory_manager.buffer.is_chunk_available(chunk_index):
            return None

        # Get chunk metadata to find item offset
        metadata = self.memory_manager.buffer._metadata.get(chunk_index)
        if not metadata or not metadata.item_offsets:
            return None

        if item_index >= len(metadata.item_offsets) - 1:
            return None

        # Calculate item boundaries
        item_start = metadata.item_offsets[item_index]
        item_end = metadata.item_offsets[item_index + 1]
        item_size = item_end - item_start

        # Get item data
        return self.memory_manager.buffer.get_partial_chunk_data(chunk_index, item_start, item_size)

    def _deserialize_item(self, item_data: bytes, item_index: int) -> Any:
        """Deserialize item data."""
        try:
            # Parse the serialized data structure
            # Format: [data_format_size][data_format][serialized_data]

            if len(item_data) < 4:
                raise ValueError("Item data too short")

            # Read data format size
            format_size = np.frombuffer(item_data[:4], np.uint32)[0]

            if len(item_data) < 4 + format_size:
                raise ValueError("Invalid data format size")

            # Read data format
            data_format = item_data[4 : 4 + format_size].decode("utf-8")

            # Read serialized data
            serialized_data = item_data[4 + format_size :]

            # Deserialize using appropriate serializer
            serializer_key = self._data_format_to_key(data_format)
            if serializer_key not in self._serializers:
                raise ValueError(f"No serializer found for format: {data_format}")

            serializer = self._serializers[serializer_key]
            return serializer.deserialize(serialized_data)

        except Exception as e:
            logger.error(f"Failed to deserialize item {item_index}: {e}")
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
