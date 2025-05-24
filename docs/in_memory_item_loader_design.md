# In-Memory Item Loader Feature Design

## Overview

This document outlines the design for an in-memory item loader feature for LitData that can read items from chunks as soon as they become available without waiting for the whole chunk to be downloaded. The feature also implements in-memory reference storage rather than writing to disk, with the ability to offload to disk if necessary, to achieve faster speeds.

## Current Architecture Analysis

### Existing System Flow
1. **Chunk Download**: Complete chunks are downloaded to disk via `PrepareChunksThread`
2. **Item Loading**: Items are loaded from downloaded chunks using `BaseItemLoader` implementations
3. **Disk Storage**: All chunks are stored on disk before items can be accessed
4. **Sequential Access**: Items can only be accessed after the entire chunk is downloaded

### Current Item Loaders
- **PyTreeLoader**: Default loader for general Python objects
- **TokensLoader**: Optimized for NLP token data with memory mapping
- **ParquetLoader**: Specialized for Parquet format with row-group level optimization

## Proposed In-Memory Item Loader

### Design Goals
1. **Streaming Access**: Read items as they become available during chunk download
2. **Memory Efficiency**: Store chunks in memory with intelligent eviction policies
3. **Disk Offloading**: Fallback to disk when memory constraints are reached
4. **Performance**: Achieve faster speeds than current disk-based approach
5. **Compatibility**: Maintain compatibility with existing item loader interface

### Architecture Components

#### 1. InMemoryItemLoader Class

```python
class InMemoryItemLoader(BaseItemLoader):
    """
    An item loader that stores chunks in memory and can stream items
    as they become available during chunk download.
    """
    
    def __init__(
        self,
        max_memory_usage: Union[int, str] = "1GB",
        eviction_policy: str = "lru",
        disk_fallback: bool = True,
        stream_threshold: float = 0.1,  # Start streaming when 10% of chunk is available
        prefetch_items: int = 10,       # Number of items to prefetch
    ):
        pass
```

#### 2. StreamingChunkBuffer

```python
class StreamingChunkBuffer:
    """
    Manages streaming chunk data as it's being downloaded.
    Allows partial reading of items before the complete chunk is available.
    """
    
    def __init__(self, chunk_info: Dict, expected_size: int):
        self.chunk_info = chunk_info
        self.expected_size = expected_size
        self.buffer = bytearray()
        self.header_parsed = False
        self.item_offsets: List[int] = []
        self.available_items: Set[int] = set()
        self.lock = threading.RLock()
    
    def append_data(self, data: bytes) -> List[int]:
        """Append new data and return newly available item indices."""
        pass
    
    def get_item(self, index: int) -> Optional[bytes]:
        """Get item if available, None otherwise."""
        pass
    
    def is_complete(self) -> bool:
        """Check if the entire chunk has been downloaded."""
        pass
```

#### 3. MemoryManager

```python
class MemoryManager:
    """
    Manages memory usage across all in-memory chunks.
    Implements eviction policies and disk offloading.
    """
    
    def __init__(
        self,
        max_memory: int,
        eviction_policy: str = "lru",
        disk_fallback_dir: Optional[str] = None
    ):
        self.max_memory = max_memory
        self.current_usage = 0
        self.chunks: Dict[int, StreamingChunkBuffer] = {}
        self.access_times: Dict[int, float] = {}
        self.disk_offloaded: Dict[int, str] = {}
    
    def add_chunk(self, chunk_index: int, buffer: StreamingChunkBuffer) -> bool:
        """Add chunk to memory, return True if successful."""
        pass
    
    def evict_chunks(self, required_space: int) -> None:
        """Evict chunks based on policy to free required space."""
        pass
    
    def offload_to_disk(self, chunk_index: int) -> str:
        """Offload chunk to disk and return file path."""
        pass
```

#### 4. StreamingDownloader

```python
class StreamingDownloader:
    """
    Enhanced downloader that can stream chunk data to the item loader
    as it's being downloaded.
    """
    
    def __init__(self, item_loader: InMemoryItemLoader):
        self.item_loader = item_loader
        self.active_downloads: Dict[int, StreamingChunkBuffer] = {}
    
    def download_chunk_streaming(
        self,
        chunk_index: int,
        chunk_info: Dict,
        on_item_available: Callable[[int, int], None]
    ) -> None:
        """Download chunk with streaming capabilities."""
        pass
```

### Key Features

#### 1. Streaming Item Access
- **Partial Parsing**: Parse chunk headers as soon as they're available
- **Progressive Loading**: Make items available as their data becomes complete
- **Non-Blocking Access**: Return available items immediately, queue requests for unavailable ones

#### 2. Memory Management
- **Smart Caching**: Keep frequently accessed chunks in memory
- **Eviction Policies**: LRU, LFU, and custom policies for memory management
- **Memory Monitoring**: Track memory usage and enforce limits
- **Disk Offloading**: Seamlessly move chunks to disk when memory is full

#### 3. Performance Optimizations
- **Prefetching**: Intelligently prefetch items based on access patterns
- **Compression Awareness**: Handle compressed chunks efficiently
- **Zero-Copy Operations**: Minimize data copying where possible
- **Async Operations**: Non-blocking download and processing

#### 4. Configuration Options
```python
@dataclass
class InMemoryConfig:
    max_memory_usage: Union[int, str] = "1GB"
    eviction_policy: str = "lru"  # lru, lfu, fifo, custom
    disk_fallback: bool = True
    disk_fallback_dir: Optional[str] = None
    stream_threshold: float = 0.1  # Start streaming at 10% download
    prefetch_items: int = 10
    max_concurrent_downloads: int = 3
    compression_aware: bool = True
```

### Implementation Strategy

#### Phase 1: Core Infrastructure
1. Implement `StreamingChunkBuffer` with partial parsing capabilities
2. Create `MemoryManager` with basic LRU eviction
3. Develop `InMemoryItemLoader` basic functionality
4. Add memory usage tracking and limits

#### Phase 2: Streaming Capabilities
1. Implement streaming download in `StreamingDownloader`
2. Add progressive item availability detection
3. Integrate with existing `PrepareChunksThread`
4. Handle chunk header parsing for different formats

#### Phase 3: Advanced Features
1. Add multiple eviction policies (LFU, FIFO, custom)
2. Implement disk offloading with seamless retrieval
3. Add compression-aware streaming
4. Implement prefetching algorithms

#### Phase 4: Optimization & Integration
1. Performance benchmarking and optimization
2. Integration with existing item loaders
3. Thread safety and concurrency improvements
4. Error handling and recovery mechanisms

### Memory Management Details

#### Eviction Policies

1. **LRU (Least Recently Used)**
   - Track access times for all chunks
   - Evict oldest accessed chunks first
   - Good general-purpose policy

2. **LFU (Least Frequently Used)**
   - Track access frequency for chunks
   - Evict least frequently accessed chunks
   - Better for workloads with clear hot/cold data

3. **Custom Policies**
   - Size-aware eviction (evict largest chunks first)
   - Dataset-aware eviction (based on chunk importance)
   - Hybrid policies combining multiple factors

#### Disk Offloading Strategy

1. **Intelligent Selection**: Offload chunks that are:
   - Completely downloaded
   - Less frequently accessed
   - Larger in size (to free more memory)

2. **Seamless Retrieval**: 
   - Keep metadata in memory
   - Load from disk transparently when accessed
   - Use memory mapping for large offloaded chunks

3. **Cleanup Management**:
   - Remove offloaded files when no longer needed
   - Implement reference counting for shared chunks
   - Clean up on process exit

### Performance Considerations

#### Memory Efficiency
- Use `bytearray` for mutable buffers during download
- Convert to `bytes` for immutable storage
- Implement copy-on-write for shared chunk data
- Use memory mapping for large chunks

#### CPU Efficiency
- Minimize serialization/deserialization overhead
- Use efficient data structures for metadata
- Implement lazy parsing where possible
- Optimize hot paths with caching

#### I/O Efficiency
- Overlap download and processing
- Use async I/O where beneficial
- Minimize disk seeks for offloaded chunks
- Batch operations when possible

### Integration Points

#### 1. StreamingDataset Integration
```python
# Enable in-memory loading
dataset = StreamingDataset(
    input_dir="s3://my-bucket/data",
    item_loader=InMemoryItemLoader(
        max_memory_usage="2GB",
        eviction_policy="lru",
        disk_fallback=True
    )
)
```

#### 2. Cache Integration
```python
# Use with Cache
cache = Cache(
    input_dir="./data",
    item_loader=InMemoryItemLoader(
        max_memory_usage="1GB",
        stream_threshold=0.05  # Start streaming at 5%
    )
)
```

#### 3. DataLoader Integration
```python
# Transparent usage with DataLoader
dataloader = StreamingDataLoader(
    dataset,
    batch_size=32,
    num_workers=4
)
```

### Error Handling & Recovery

#### Download Failures
- Retry failed chunk downloads
- Fallback to disk-based loading on persistent failures
- Graceful degradation of streaming capabilities

#### Memory Pressure
- Emergency eviction when memory limits exceeded
- Pause downloads during high memory pressure
- Alert users about memory constraints

#### Corruption Detection
- Validate chunk integrity during streaming
- Detect and handle partial corruptions
- Provide recovery mechanisms for damaged chunks

### Monitoring & Debugging

#### Metrics Collection
- Memory usage statistics
- Cache hit/miss ratios
- Download and processing times
- Eviction frequency and reasons

#### Debug Capabilities
- Chunk state visualization
- Memory usage tracking
- Performance profiling hooks
- Detailed logging options

### Benchmarking Plan

#### Performance Metrics
1. **Latency**: Time to first item availability
2. **Throughput**: Items processed per second
3. **Memory Efficiency**: Memory usage vs. performance trade-off
4. **Cache Effectiveness**: Hit ratio and eviction frequency

#### Test Scenarios
1. **Small Chunks**: Many small chunks with high access frequency
2. **Large Chunks**: Few large chunks with selective access
3. **Mixed Workload**: Combination of access patterns
4. **Memory Constrained**: Limited memory scenarios
5. **Network Variations**: Different download speeds and reliability

#### Comparison Baselines
- Current disk-based PyTreeLoader
- Current TokensLoader with memory mapping
- Raw in-memory storage without streaming
- Third-party streaming solutions

### Future Extensions

#### 1. Compression Streaming
- Support for streaming decompression
- Format-specific optimizations (zstd, gzip, etc.)
- Adaptive compression based on network speed

#### 2. Distributed Caching
- Share cached chunks across workers
- Implement distributed eviction policies
- Add network-aware caching strategies

#### 3. ML-Driven Optimization
- Learn access patterns for better prefetching
- Adaptive eviction based on workload characteristics
- Automatic parameter tuning

#### 4. Advanced Streaming Protocols
- Support for HTTP range requests
- WebSocket-based streaming
- Custom streaming protocols for cloud providers

## Conclusion

The In-Memory Item Loader feature will significantly improve LitData's performance by:

1. **Reducing Latency**: Items become available as soon as their data is downloaded
2. **Improving Throughput**: Eliminate disk I/O bottlenecks for frequently accessed data
3. **Better Resource Utilization**: Intelligent memory management with disk fallback
4. **Maintaining Compatibility**: Seamless integration with existing APIs

The phased implementation approach ensures incremental value delivery while maintaining system stability. The comprehensive monitoring and benchmarking plan will validate performance improvements and guide optimization efforts.
