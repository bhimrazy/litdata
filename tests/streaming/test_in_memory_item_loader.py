"""Tests for in-memory item loader functionality."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from litdata.streaming.in_memory_item_loader import InMemoryItemLoader
from litdata.streaming.streaming_buffer import EvictionPolicy, StreamingChunkBuffer, StreamingConfig
from litdata.streaming.streaming_downloader import StreamingDownloaderManager


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config():
    return StreamingConfig(
        max_memory_size=1024 * 1024,  # 1MB
        eviction_policy=EvictionPolicy.LRU,
        chunk_timeout=30,
        enable_memory_monitoring=True,
    )


@pytest.fixture
def streaming_buffer(sample_config):
    return StreamingChunkBuffer(sample_config)


@pytest.fixture
def mock_downloader_manager():
    with patch("litdata.streaming.in_memory_item_loader.StreamingDownloaderManager") as mock:
        yield mock


class TestStreamingChunkBuffer:
    """Test the streaming chunk buffer."""

    def test_buffer_initialization(self, sample_config):
        buffer = StreamingChunkBuffer(sample_config)
        assert buffer.config == sample_config
        assert buffer._current_memory_usage == 0
        assert len(buffer._chunks) == 0

    def test_store_chunk_data(self, streaming_buffer):
        chunk_data = b"test data"
        success = streaming_buffer.store_chunk_data(0, chunk_data)
        assert success
        assert streaming_buffer.is_chunk_available(0)
        assert streaming_buffer.get_chunk_data(0) == chunk_data

    def test_append_chunk_data(self, streaming_buffer):
        # Store initial data
        initial_data = b"initial"
        streaming_buffer.store_chunk_data(0, initial_data, is_complete=False)

        # Append more data
        additional_data = b" additional"
        success = streaming_buffer.append_chunk_data(0, additional_data)
        assert success

        # Check combined data
        combined_data = streaming_buffer.get_chunk_data(0)
        assert combined_data == initial_data + additional_data

    def test_memory_eviction(self, sample_config):
        # Create buffer with very small memory limit
        small_config = StreamingConfig(
            max_memory_size=100,  # 100 bytes
            eviction_policy=EvictionPolicy.LRU,
        )
        buffer = StreamingChunkBuffer(small_config)

        # Store data that exceeds memory limit
        large_data = b"x" * 60  # 60 bytes
        buffer.store_chunk_data(0, large_data)
        buffer.store_chunk_data(1, large_data)  # This should succeed

        # Store more data to trigger eviction
        more_data = b"y" * 60
        success = buffer.store_chunk_data(2, more_data)

        # Should evict oldest chunk (0) to make room
        assert not buffer.is_chunk_available(0)
        assert buffer.is_chunk_available(1)
        assert buffer.is_chunk_available(2)

    def test_eviction_policies(self, sample_config):
        # Test FIFO eviction
        fifo_config = StreamingConfig(
            max_memory_size=100,
            eviction_policy=EvictionPolicy.FIFO,
        )
        buffer = StreamingChunkBuffer(fifo_config)

        data = b"x" * 40
        buffer.store_chunk_data(0, data)
        buffer.store_chunk_data(1, data)

        # Access chunk 0 to make it more recently used
        buffer.get_chunk_data(0)

        # Store more data - with FIFO, chunk 0 should still be evicted
        more_data = b"y" * 40
        buffer.store_chunk_data(2, more_data)

        # With FIFO, the first stored chunk should be evicted regardless of access
        assert not buffer.is_chunk_available(0)


class TestInMemoryItemLoader:
    """Test the in-memory item loader."""

    def test_loader_initialization(self, mock_downloader_manager):
        loader = InMemoryItemLoader()
        assert loader.streaming_config is not None
        assert loader.chunk_buffer is not None
        assert loader.downloader_manager is not None

    def test_generate_intervals(self):
        loader = InMemoryItemLoader()

        # Mock chunk info
        mock_chunks = [
            {"filename": "chunk_0.bin", "dim": 10},
            {"filename": "chunk_1.bin", "dim": 15},
        ]

        intervals = loader.generate_intervals(mock_chunks)

        # Should generate intervals for each chunk
        assert len(intervals) == 2
        assert intervals[0].chunk_index == 0
        assert intervals[0].start == 0
        assert intervals[0].stop == 10
        assert intervals[1].chunk_index == 1
        assert intervals[1].start == 10
        assert intervals[1].stop == 25

    def test_is_item_available(self, mock_downloader_manager):
        loader = InMemoryItemLoader()

        # Store some test data
        test_data = b"test chunk data"
        loader.chunk_buffer.store_chunk_data(0, test_data)

        # Mock chunk metadata
        loader.chunk_buffer._metadata[0].item_count = 5

        # Test item availability
        assert loader.is_item_available(0, 0)  # chunk 0, item 0
        assert loader.is_item_available(0, 4)  # chunk 0, item 4
        assert not loader.is_item_available(0, 5)  # item beyond chunk
        assert not loader.is_item_available(1, 0)  # chunk not available

    def test_memory_stats(self, mock_downloader_manager):
        loader = InMemoryItemLoader()

        # Store some test data
        test_data = b"test chunk data"
        loader.chunk_buffer.store_chunk_data(0, test_data)

        stats = loader.get_memory_stats()
        assert "total_memory_usage" in stats
        assert "max_memory_size" in stats
        assert "num_chunks_cached" in stats
        assert "memory_utilization" in stats

        assert stats["total_memory_usage"] == len(test_data)
        assert stats["num_chunks_cached"] == 1

    @patch("litdata.streaming.in_memory_item_loader.logger")
    def test_delete_cleanup(self, mock_logger, mock_downloader_manager):
        loader = InMemoryItemLoader()

        # Store some test data
        test_data = b"test chunk data"
        loader.chunk_buffer.store_chunk_data(0, test_data)

        # Delete should clear all data
        loader.delete()

        assert loader.chunk_buffer._current_memory_usage == 0
        assert len(loader.chunk_buffer._chunks) == 0
        mock_logger.info.assert_called()


class TestStreamingDownloaderManager:
    """Test the streaming downloader manager."""

    def test_manager_initialization(self):
        manager = StreamingDownloaderManager()
        assert len(manager._downloaders) > 0  # Should have at least local downloader

    def test_get_downloader_s3(self):
        manager = StreamingDownloaderManager()
        downloader = manager.get_downloader("s3://bucket/file.bin")
        assert downloader is not None
        assert downloader.supports_url("s3://bucket/file.bin")

    def test_get_downloader_local(self):
        manager = StreamingDownloaderManager()
        downloader = manager.get_downloader("/local/path/file.bin")
        assert downloader is not None
        assert downloader.supports_url("/local/path/file.bin")

    def test_get_downloader_unsupported(self):
        manager = StreamingDownloaderManager()
        downloader = manager.get_downloader("unsupported://protocol/file.bin")
        assert downloader is None

    def test_stream_chunk_local_file(self, temp_dir, sample_config):
        # Create a test file
        test_file = os.path.join(temp_dir, "test.bin")
        test_data = b"Hello, streaming world!"
        with open(test_file, "wb") as f:
            f.write(test_data)

        # Test streaming
        manager = StreamingDownloaderManager()
        buffer = StreamingChunkBuffer(sample_config)

        success = manager.stream_chunk(test_file, buffer, 0)
        assert success
        assert buffer.is_chunk_available(0)
        assert buffer.get_chunk_data(0) == test_data


class TestIntegration:
    """Integration tests for the complete streaming system."""

    def test_end_to_end_local_streaming(self, temp_dir):
        # Create a test chunk file with binary data
        test_file = os.path.join(temp_dir, "chunk_0.bin")

        # Create mock chunk data with header
        num_items = 3
        item_data = [b"item1", b"item2", b"item3"]

        # Calculate offsets
        offsets = []
        current_offset = 8 + (num_items * 8)  # header + offset table
        for item in item_data:
            offsets.append(current_offset)
            current_offset += len(item)

        # Write chunk file
        with open(test_file, "wb") as f:
            # Write header
            f.write(num_items.to_bytes(8, "little"))

            # Write offset table
            for offset in offsets:
                f.write(offset.to_bytes(8, "little"))

            # Write item data
            for item in item_data:
                f.write(item)

        # Test the loader
        loader = InMemoryItemLoader()

        # Pre-load the chunk
        success = loader.pre_load_chunk(test_file, 0)
        assert success

        # Verify we can load items
        assert loader.is_item_available(0, 0)
        assert loader.is_item_available(0, 1)
        assert loader.is_item_available(0, 2)
        assert not loader.is_item_available(0, 3)

        # Test memory stats
        stats = loader.get_memory_stats()
        assert stats["num_chunks_cached"] == 1
        assert stats["total_memory_usage"] > 0

    def test_memory_pressure_handling(self, temp_dir):
        # Create small memory config
        small_config = StreamingConfig(
            max_memory_size=200,  # Very small
            eviction_policy=EvictionPolicy.LRU,
        )

        loader = InMemoryItemLoader(streaming_config=small_config)

        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(temp_dir, f"chunk_{i}.bin")
            test_data = b"x" * 80  # Large enough to trigger eviction
            with open(test_file, "wb") as f:
                f.write(test_data)
            test_files.append(test_file)

        # Load chunks - should trigger eviction
        for i, test_file in enumerate(test_files):
            loader.pre_load_chunk(test_file, i)

        # Check that old chunks were evicted
        stats = loader.get_memory_stats()
        assert stats["num_chunks_cached"] < 3  # Some chunks should be evicted
