"""
InMemoryItemLoader Usage Examples and Documentation
=================================================

This document provides comprehensive examples and documentation for using
the InMemoryItemLoader with LitData for high-performance streaming data access.

Overview
--------
The InMemoryItemLoader is designed to provide faster data access by keeping
chunks in memory, enabling progressive downloading and streaming of data
directly from cloud storage without waiting for complete chunks to download.

Key Features:
- In-memory chunk caching with configurable memory limits
- Progressive streaming from cloud storage (S3, GCS, Azure)
- LRU eviction policy for memory management
- Automatic fallback to direct file access when needed
- Compatible with existing LitData APIs

Installation
-----------
No additional dependencies are required beyond the standard LitData installation.
The InMemoryItemLoader is available in LitData and can be imported directly:

```python
from litdata import InMemoryItemLoader, StreamingDataset, StreamingConfig
from litdata.streaming.streaming_buffer import EvictionPolicy
```

Basic Usage
-----------

### Example 1: Local Dataset with In-Memory Loading

```python
import torch
from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import StreamingConfig, EvictionPolicy

# Configure streaming parameters
streaming_config = StreamingConfig(
    max_memory_size=100 * 1024 * 1024,  # 100MB memory limit
    eviction_policy=EvictionPolicy.LRU,
    chunk_timeout=30.0,
    enable_memory_monitoring=True,
)

# Create in-memory item loader
loader = InMemoryItemLoader(streaming_config=streaming_config)

# Use with StreamingDataset
dataset = StreamingDataset(
    input_dir="path/to/optimized/dataset",
    item_loader=loader
)

# Access data normally - caching happens automatically
for i in range(10):
    sample = dataset[i]
    print(f"Sample {i}: {type(sample)}")

# Check memory usage
stats = loader.get_memory_stats()
print(f"Memory usage: {stats['memory_utilization']:.1%}")
```

### Example 2: S3 Dataset with Unsigned Access

```python
from botocore import UNSIGNED
from botocore.config import Config
from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import StreamingConfig, EvictionPolicy

# Configure S3 unsigned access
storage_options = {
    "config": Config(signature_version=UNSIGNED)
}

# Configure for larger memory usage
streaming_config = StreamingConfig(
    max_memory_size=500 * 1024 * 1024,  # 500MB
    eviction_policy=EvictionPolicy.LRU,
    chunk_timeout=60.0,
    enable_memory_monitoring=True,
)

loader = InMemoryItemLoader(streaming_config=streaming_config)

# Public S3 dataset
dataset = StreamingDataset(
    input_dir="s3://public-dataset/optimized-data",
    item_loader=loader,
    storage_options=storage_options
)

# Multiple epoch training benefits from caching
for epoch in range(3):
    print(f"Epoch {epoch + 1}")
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        # Training step here...
    
    # Monitor memory after each epoch
    stats = loader.get_memory_stats()
    print(f"  Memory: {stats['memory_utilization']:.1%} "
          f"({stats['num_chunks_cached']} chunks cached)")
```

### Example 3: Memory-Constrained Environment

```python
from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import StreamingConfig, EvictionPolicy

# Conservative memory settings for limited environments
streaming_config = StreamingConfig(
    max_memory_size=50 * 1024 * 1024,  # Only 50MB
    eviction_policy=EvictionPolicy.LRU,
    chunk_timeout=30.0,
    enable_memory_monitoring=True,
)

loader = InMemoryItemLoader(streaming_config=streaming_config)

dataset = StreamingDataset(
    input_dir="path/to/large/dataset",
    item_loader=loader
)

# The loader will automatically evict old chunks and fallback as needed
for i in range(1000):
    sample = dataset[i]
    
    if i % 100 == 0:
        stats = loader.get_memory_stats()
        print(f"Processed {i} samples, memory usage: {stats['memory_utilization']:.1%}")
```

Configuration Options
-------------------

### StreamingConfig Parameters

- **max_memory_size** (int): Maximum memory to use for caching chunks (in bytes)
- **eviction_policy** (EvictionPolicy): Strategy for removing chunks from memory
  - `EvictionPolicy.LRU`: Least Recently Used (recommended)
- **chunk_timeout** (float): Maximum time to wait for chunk streaming (seconds)
- **enable_memory_monitoring** (bool): Enable detailed memory usage tracking

### EvictionPolicy Options

Currently supported:
- `EvictionPolicy.LRU`: Removes least recently used chunks when memory limit is reached

Performance Considerations
------------------------

### When to Use InMemoryItemLoader

**Best Use Cases:**
- Multi-epoch training where data is accessed repeatedly
- Random access patterns during training
- Cloud storage with high latency but good bandwidth
- Datasets with small to medium-sized chunks

**Less Optimal Cases:**
- Single-pass sequential access
- Very large chunks that exceed memory limits
- Extremely memory-constrained environments

### Memory Management

The InMemoryItemLoader uses a sophisticated memory management system:

1. **Progressive Loading**: Chunks are streamed and items become available as data arrives
2. **LRU Eviction**: When memory limit is reached, least recently used chunks are evicted
3. **Graceful Fallback**: If streaming fails, the loader falls back to direct file access
4. **Memory Monitoring**: Optional detailed tracking of memory usage and chunk statistics

### Performance Tips

1. **Set Appropriate Memory Limits**: Use 60-80% of available memory for optimal performance
2. **Monitor Memory Usage**: Use `get_memory_stats()` to understand caching behavior
3. **Consider Chunk Size**: Smaller chunks provide better granular caching control
4. **Repeated Access**: Maximum benefit comes from accessing the same data multiple times

API Reference
-----------

### InMemoryItemLoader

```python
class InMemoryItemLoader(BaseItemLoader):
    def __init__(self, streaming_config: Optional[StreamingConfig] = None):
        """Initialize the in-memory item loader.
        
        Args:
            streaming_config: Configuration for streaming behavior
        """
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics.
        
        Returns:
            Dictionary with keys:
            - total_memory_usage: Current memory usage in bytes
            - max_memory_size: Maximum allowed memory in bytes  
            - num_chunks_cached: Number of chunks currently in memory
            - memory_utilization: Memory usage as a fraction (0.0-1.0)
            - chunks: Per-chunk statistics
        """
```

### StreamingConfig

```python
@dataclass
class StreamingConfig:
    max_memory_size: int = 100 * 1024 * 1024  # 100MB default
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    chunk_timeout: float = 30.0  # seconds
    enable_memory_monitoring: bool = True
```

Troubleshooting
--------------

### Common Issues

1. **High Memory Usage**
   - Reduce `max_memory_size` in StreamingConfig
   - Monitor with `get_memory_stats()` to understand usage patterns

2. **Slow Performance on First Access**
   - This is expected as data is being streamed and cached
   - Subsequent accesses to the same chunks will be much faster

3. **Fallback to Direct File Access**
   - Occurs when streaming fails or memory is insufficient
   - Check logs for "Streaming failed" messages
   - Consider increasing memory limits or improving network connectivity

4. **Import Errors**
   ```python
   # Correct import:
   from litdata import InMemoryItemLoader
   
   # Also available:
   from litdata.streaming import InMemoryItemLoader
   ```

Examples with Different Cloud Providers
-------------------------------------

### Google Cloud Storage

```python
from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import StreamingConfig

streaming_config = StreamingConfig(max_memory_size=200 * 1024 * 1024)
loader = InMemoryItemLoader(streaming_config=streaming_config)

# If using service account authentication, configure storage_options:
storage_options = {
    "token": "/path/to/service-account.json"
}

dataset = StreamingDataset(
    input_dir="gs://my-bucket/optimized-dataset",
    item_loader=loader,
    storage_options=storage_options
)
```

### Azure Blob Storage

```python
from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import StreamingConfig

streaming_config = StreamingConfig(max_memory_size=200 * 1024 * 1024)
loader = InMemoryItemLoader(streaming_config=streaming_config)

# Configure Azure authentication
storage_options = {
    "account_name": "mystorageaccount",
    "account_key": "myaccountkey"
}

dataset = StreamingDataset(
    input_dir="azure://mycontainer/optimized-dataset",
    item_loader=loader,
    storage_options=storage_options
)
```

Benchmarking and Performance Testing
----------------------------------

For performance testing, use the following pattern:

```python
import time
from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import StreamingConfig

def benchmark_loader(dataset, num_samples=100, num_epochs=3):
    """Benchmark repeated access performance."""
    
    # First pass (cold cache)
    start_time = time.time()
    for i in range(num_samples):
        _ = dataset[i]
    cold_time = time.time() - start_time
    
    # Subsequent passes (warm cache)
    warm_times = []
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in range(num_samples):
            _ = dataset[i]
        warm_times.append(time.time() - start_time)
    
    avg_warm_time = sum(warm_times) / len(warm_times)
    speedup = cold_time / avg_warm_time
    
    print(f"Cold cache time: {cold_time:.2f}s")
    print(f"Warm cache time: {avg_warm_time:.2f}s") 
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup

# Usage
streaming_config = StreamingConfig(max_memory_size=200 * 1024 * 1024)
loader = InMemoryItemLoader(streaming_config=streaming_config)
dataset = StreamingDataset("path/to/data", item_loader=loader)

speedup = benchmark_loader(dataset)
```

This benchmarking approach will help you understand the performance characteristics
of the InMemoryItemLoader for your specific use case and data patterns.
"""
