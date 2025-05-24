#!/usr/bin/env python3
"""
Simple InMemoryItemLoader Example

This example demonstrates basic usage of the InMemoryItemLoader
for faster streaming data access.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "../src"))

from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import StreamingConfig, EvictionPolicy


def basic_usage_example():
    """Demonstrate basic InMemoryItemLoader usage."""
    print("🚀 InMemoryItemLoader Basic Usage Example")
    print("=" * 45)

    # Configure streaming parameters
    streaming_config = StreamingConfig(
        max_memory_size=100 * 1024 * 1024,  # 100MB
        eviction_policy=EvictionPolicy.LRU,
        chunk_timeout=30.0,
        enable_memory_monitoring=True,
    )

    # Create in-memory item loader
    loader = InMemoryItemLoader(streaming_config=streaming_config)
    print(f"✅ Created InMemoryItemLoader with {streaming_config.max_memory_size // (1024 * 1024)}MB limit")

    # For this example, you would use your own optimized dataset path
    # dataset = StreamingDataset(
    #     input_dir="path/to/your/optimized/dataset",
    #     item_loader=loader
    # )

    print("\n📖 Example usage:")
    print("""
# Create dataset with in-memory loader
from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import StreamingConfig

# Configure streaming
config = StreamingConfig(max_memory_size=100 * 1024 * 1024)  # 100MB
loader = InMemoryItemLoader(streaming_config=config)

# Use with your dataset
dataset = StreamingDataset(
    input_dir="path/to/optimized/dataset",
    item_loader=loader
)

# Access data - caching happens automatically
for epoch in range(3):
    for i in range(len(dataset)):
        sample = dataset[i]
        # Training step here...
    
    # Check memory usage
    stats = loader.get_memory_stats()
    print(f"Memory usage: {stats['memory_utilization']:.1%}")
""")

    print("\n💡 Key Benefits:")
    print("• Faster access to frequently used data")
    print("• Automatic memory management with LRU eviction")
    print("• Progressive streaming from cloud storage")
    print("• Graceful fallback when memory is insufficient")

    print("\n📊 Memory Stats API:")
    stats = loader.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return True


def cloud_storage_example():
    """Show how to use with cloud storage."""
    print("\n☁️ Cloud Storage Integration")
    print("=" * 30)

    print("""
# S3 Example with unsigned access
from botocore import UNSIGNED
from botocore.config import Config

storage_options = {"config": Config(signature_version=UNSIGNED)}

dataset = StreamingDataset(
    input_dir="s3://your-bucket/optimized-dataset",
    item_loader=InMemoryItemLoader(streaming_config=config),
    storage_options=storage_options
)

# Google Cloud Storage
storage_options = {"token": "/path/to/service-account.json"}
dataset = StreamingDataset(
    input_dir="gs://your-bucket/optimized-dataset", 
    item_loader=loader,
    storage_options=storage_options
)

# Azure Blob Storage  
storage_options = {
    "account_name": "youraccount",
    "account_key": "yourkey"
}
dataset = StreamingDataset(
    input_dir="azure://container/optimized-dataset",
    item_loader=loader, 
    storage_options=storage_options
)
""")


def performance_tips():
    """Share performance optimization tips."""
    print("\n⚡ Performance Tips")
    print("=" * 20)

    print("""
1. Set Memory Limit Appropriately:
   • Use 60-80% of available memory for optimal performance
   • Monitor with get_memory_stats() to understand usage

2. Multi-Epoch Training Benefits Most:
   • First epoch streams and caches data
   • Subsequent epochs benefit from cached chunks
   • Random access patterns see biggest speedup

3. Chunk Size Considerations:
   • Smaller chunks = better granular control
   • Larger chunks = fewer cache misses
   • Balance based on your access patterns

4. Memory Management:
   • LRU eviction removes oldest unused chunks
   • Automatic fallback ensures reliability
   • Monitor memory_utilization for insights

5. Cloud Storage Optimization:
   • Use appropriate storage_options for authentication
   • Consider bandwidth vs. latency tradeoffs
   • Larger memory limits help with network latency
""")


def main():
    """Run the example."""
    try:
        basic_usage_example()
        cloud_storage_example()
        performance_tips()

        print("\n🎉 Example completed!")
        print("\nNext steps:")
        print("1. Try with your own optimized dataset")
        print("2. Experiment with different memory limits")
        print("3. Monitor performance with get_memory_stats()")
        print("4. Check out the full documentation in docs/")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
