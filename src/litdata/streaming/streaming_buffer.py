"""Streaming buffer components for in-memory chunk management."""

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("litdata.streaming.streaming_buffer")


class EvictionPolicy(Enum):
    """Eviction policies for memory management."""

    LRU = "lru"
    FIFO = "fifo"
    SIZE_BASED = "size_based"


@dataclass
class ChunkMetadata:
    """Metadata about a chunk in the buffer."""

    chunk_index: int
    size: int
    access_count: int
    last_access_time: float
    is_complete: bool
    num_items: int
    item_offsets: Optional[List[int]] = None


@dataclass
class StreamingConfig:
    """Configuration for streaming buffer."""

    max_memory_size: int = 1024 * 1024 * 1024  # 1GB default
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_disk_offload: bool = True
    disk_offload_threshold: float = 0.8  # Offload when 80% memory used
    prefetch_items: int = 10  # Number of items to prefetch ahead
    enable_compression: bool = False
    compression_threshold: int = 1024  # Compress chunks larger than 1KB


class StreamingChunkBuffer:
    """Manages in-memory buffering of chunk data with streaming capabilities.

    Supports partial chunk loading, progressive item availability, and
    memory-efficient operations.
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self._chunks: Dict[int, bytes] = {}
        self._metadata: Dict[int, ChunkMetadata] = {}
        self._access_order = OrderedDict()  # For LRU tracking
        self._lock = threading.RLock()
        self._current_memory_usage = 0

    def get_chunk_data(self, chunk_index: int) -> Optional[bytes]:
        """Get complete chunk data if available."""
        with self._lock:
            if chunk_index in self._chunks:
                self._update_access(chunk_index)
                return self._chunks[chunk_index]
            return None

    def get_partial_chunk_data(self, chunk_index: int, offset: int, size: int) -> Optional[bytes]:
        """Get partial chunk data from the specified offset."""
        with self._lock:
            if chunk_index in self._chunks:
                chunk_data = self._chunks[chunk_index]
                if offset + size <= len(chunk_data):
                    self._update_access(chunk_index)
                    return chunk_data[offset : offset + size]
            return None

    def store_chunk_data(self, chunk_index: int, data: bytes, is_complete: bool = True) -> bool:
        """Store chunk data in buffer.

        Args:
            chunk_index: Index of the chunk
            data: Chunk data to store
            is_complete: Whether this is the complete chunk or partial data

        Returns:
            True if stored successfully, False if evicted due to memory constraints
        """
        with self._lock:
            # Check if we need to evict data
            data_size = len(data)
            if self._current_memory_usage + data_size > self.config.max_memory_size and not self._evict_chunks(
                data_size
            ):
                # Could not free enough space
                return False

            # Store the chunk data
            if chunk_index in self._chunks:
                # Update existing chunk
                old_size = len(self._chunks[chunk_index])
                self._current_memory_usage -= old_size

            self._chunks[chunk_index] = data
            self._current_memory_usage += data_size

            # Parse chunk metadata
            metadata = self._parse_chunk_metadata(chunk_index, data, is_complete)
            self._metadata[chunk_index] = metadata

            # Update access order
            self._update_access(chunk_index)

            logger.debug(f"Stored chunk {chunk_index}, size: {data_size}, total memory: {self._current_memory_usage}")
            return True

    def append_chunk_data(self, chunk_index: int, data: bytes) -> bool:
        """Append data to an existing partial chunk.

        Args:
            chunk_index: Index of the chunk
            data: Additional data to append

        Returns:
            True if appended successfully
        """
        with self._lock:
            if chunk_index not in self._chunks:
                # If chunk doesn't exist, treat as new chunk
                return self.store_chunk_data(chunk_index, data, is_complete=False)

            # Check memory constraints
            additional_size = len(data)
            if self._current_memory_usage + additional_size > self.config.max_memory_size:  # noqa: SIM102
                if not self._evict_chunks(additional_size):
                    return False

            # Append data
            existing_data = self._chunks[chunk_index]
            new_data = existing_data + data

            self._chunks[chunk_index] = new_data
            self._current_memory_usage += additional_size

            # Update metadata
            metadata = self._metadata.get(chunk_index)
            if metadata:
                metadata.size = len(new_data)
                # Re-parse to check if chunk is now complete
                self._metadata[chunk_index] = self._parse_chunk_metadata(chunk_index, new_data, False)

            self._update_access(chunk_index)
            return True

    def is_chunk_available(self, chunk_index: int) -> bool:
        """Check if chunk is available in buffer."""
        return chunk_index in self._chunks

    def is_chunk_complete(self, chunk_index: int) -> bool:
        """Check if chunk is completely loaded."""
        metadata = self._metadata.get(chunk_index)
        return metadata.is_complete if metadata else False

    def get_available_items(self, chunk_index: int) -> List[int]:
        """Get list of item indices that are available in the chunk."""
        metadata = self._metadata.get(chunk_index)
        if not metadata or not metadata.item_offsets:
            return []

        chunk_data = self._chunks.get(chunk_index)
        if not chunk_data:
            return []

        available_items = []
        for i, offset in enumerate(metadata.item_offsets[:-1]):  # Exclude last offset
            next_offset = metadata.item_offsets[i + 1]
            # Check if we have enough data for this item
            if next_offset <= len(chunk_data):
                available_items.append(i)

        return available_items

    def remove_chunk(self, chunk_index: int) -> bool:
        """Remove chunk from buffer."""
        with self._lock:
            if chunk_index in self._chunks:
                chunk_size = len(self._chunks[chunk_index])
                del self._chunks[chunk_index]
                del self._metadata[chunk_index]
                self._access_order.pop(chunk_index, None)
                self._current_memory_usage -= chunk_size
                return True
            return False

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        with self._lock:
            return {
                "current_usage": self._current_memory_usage,
                "max_size": self.config.max_memory_size,
                "usage_ratio": self._current_memory_usage / self.config.max_memory_size,
                "num_chunks": len(self._chunks),
                "chunks": {
                    idx: {"size": len(data), "complete": self._metadata[idx].is_complete}
                    for idx, data in self._chunks.items()
                },
            }

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._chunks.clear()
            self._metadata.clear()
            self._access_order.clear()
            self._current_memory_usage = 0

    def _parse_chunk_metadata(self, chunk_index: int, data: bytes, is_complete: bool) -> ChunkMetadata:
        """Parse chunk metadata from binary data.

        Chunk binary format:
        +------------+---------------+-------------+
        | num_items  | offset_array  | item_data   |
        +------------+---------------+-------------+
        | uint32     | uint32[N+1]   | bytes       |
        | 4 bytes    | 4*(N+1) bytes | variable    |
        +------------+---------------+-------------+
        """
        import time

        metadata = ChunkMetadata(
            chunk_index=chunk_index,
            size=len(data),
            access_count=0,
            last_access_time=time.time(),
            is_complete=is_complete,
            num_items=0,
            item_offsets=None,
        )

        # Try to parse chunk structure if we have enough data
        if len(data) >= 4:
            try:
                # Read number of items (first 4 bytes)
                num_items = np.frombuffer(data[:4], np.uint32)[0]
                metadata.num_items = num_items

                # Calculate expected offset array size
                offset_array_size = (num_items + 1) * 4  # +1 for end offset
                header_size = 4 + offset_array_size

                if len(data) >= header_size:
                    # Read offset array
                    offset_data = data[4:header_size]
                    offsets = np.frombuffer(offset_data, np.uint32).tolist()
                    metadata.item_offsets = offsets

                    # Check if chunk is complete by verifying last offset
                    if len(offsets) > 0:
                        expected_total_size = offsets[-1]
                        metadata.is_complete = len(data) >= expected_total_size

            except Exception as e:
                logger.debug(f"Could not parse chunk metadata for chunk {chunk_index}: {e}")

        return metadata

    def _update_access(self, chunk_index: int) -> None:
        """Update access tracking for LRU."""
        import time

        # Update LRU order
        self._access_order.pop(chunk_index, None)
        self._access_order[chunk_index] = True

        # Update metadata
        if chunk_index in self._metadata:
            metadata = self._metadata[chunk_index]
            metadata.access_count += 1
            metadata.last_access_time = time.time()

    def _evict_chunks(self, required_space: int) -> bool:
        """Evict chunks based on configured policy to free required space.

        Args:
            required_space: Minimum space needed to free

        Returns:
            True if enough space was freed
        """
        freed_space = 0
        chunks_to_evict = []

        if self.config.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used chunks
            for chunk_index in list(self._access_order.keys()):
                chunk_size = len(self._chunks[chunk_index])
                chunks_to_evict.append(chunk_index)
                freed_space += chunk_size

                if freed_space >= required_space:
                    break

        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            # Evict oldest chunks first
            sorted_chunks = sorted(self._metadata.items(), key=lambda x: x[1].last_access_time)

            for chunk_index, metadata in sorted_chunks:
                chunks_to_evict.append(chunk_index)
                freed_space += metadata.size

                if freed_space >= required_space:
                    break

        elif self.config.eviction_policy == EvictionPolicy.SIZE_BASED:
            # Evict largest chunks first
            sorted_chunks = sorted(self._metadata.items(), key=lambda x: x[1].size, reverse=True)

            for chunk_index, metadata in sorted_chunks:
                chunks_to_evict.append(chunk_index)
                freed_space += metadata.size

                if freed_space >= required_space:
                    break

        # Perform eviction
        for chunk_index in chunks_to_evict:
            self.remove_chunk(chunk_index)
            logger.debug(f"Evicted chunk {chunk_index} to free memory")

        return freed_space >= required_space


class MemoryManager:
    """Manages memory allocation and eviction policies for streaming data.

    Provides high-level memory management operations and monitoring.
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.buffer = StreamingChunkBuffer(config)
        self._lock = threading.Lock()

    def should_offload_to_disk(self) -> bool:
        """Check if we should start offloading data to disk."""
        usage = self.buffer.get_memory_usage()
        return usage["usage_ratio"] >= self.config.disk_offload_threshold

    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)."""
        usage = self.buffer.get_memory_usage()
        return usage["usage_ratio"]

    def reserve_memory(self, size: int) -> bool:
        """Try to reserve memory for a new allocation.

        Args:
            size: Size in bytes to reserve

        Returns:
            True if memory can be reserved
        """
        with self._lock:
            current_usage = self.buffer._current_memory_usage
            if current_usage + size <= self.config.max_memory_size:
                return True

            # Try to free space through eviction
            required_space = (current_usage + size) - self.config.max_memory_size
            return self.buffer._evict_chunks(required_space)

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by applying compression and eviction.

        Returns:
            Statistics about the optimization
        """
        stats = {"chunks_compressed": 0, "chunks_evicted": 0, "memory_freed": 0, "compression_ratio": 1.0}

        # TODO: Implement compression if enabled
        if self.config.enable_compression:
            # Compress large chunks to save memory
            pass

        # Perform eviction if memory pressure is high
        if self.get_memory_pressure() > 0.9:
            # Evict some chunks to reduce pressure
            target_reduction = int(0.2 * self.config.max_memory_size)  # Free 20%
            if self.buffer._evict_chunks(target_reduction):
                stats["memory_freed"] = target_reduction

        return stats
