#!/usr/bin/env python3
"""
Enhanced Performance Test for InMemoryItemLoader

This script demonstrates optimized usage patterns and compares performance
between standard and in-memory loaders under realistic scenarios.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from botocore import UNSIGNED
from botocore.config import Config

from litdata import InMemoryItemLoader, StreamingDataset
from litdata.streaming.streaming_buffer import EvictionPolicy, StreamingConfig


def create_optimized_config(memory_size_mb: int = 200) -> StreamingConfig:
    """Create optimized streaming configuration."""
    return StreamingConfig(
        max_memory_size=memory_size_mb * 1024 * 1024,
        eviction_policy=EvictionPolicy.LRU,
        chunk_timeout=60.0,  # Longer timeout for better stability
        enable_memory_monitoring=True,
    )


def benchmark_repeated_access():
    """Test performance with repeated access patterns (multi-epoch training simulation)."""
    print("🏃‍♂️ Benchmarking Repeated Access Performance")
    print("=" * 50)
    
    # Configure unsigned S3 access
    storage_options = {"config": Config(signature_version=UNSIGNED)}
    
    # Test parameters
    test_samples = 50  # Focused test on fewer samples for clearer results
    num_epochs = 5     # More epochs to show caching benefits
    
    # Test 1: Standard loader
    print("📊 Testing Standard Loader...")
    start_time = time.time()
    
    standard_dataset = StreamingDataset(
        input_dir="s3://pl-flash-data/optimized_tiny_imagenet",
        storage_options=storage_options
    )
    
    standard_times = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for i in range(min(test_samples, len(standard_dataset))):
            _ = standard_dataset[i]
        epoch_time = time.time() - epoch_start
        standard_times.append(epoch_time)
        print(f"  Epoch {epoch + 1}: {epoch_time:.2f}s")
    
    standard_total = time.time() - start_time
    standard_avg = sum(standard_times) / len(standard_times)
    
    # Test 2: In-memory loader
    print("\n🧠 Testing InMemory Loader...")
    streaming_config = create_optimized_config(300)  # 300MB for better caching
    
    start_time = time.time()
    inmemory_dataset = StreamingDataset(
        input_dir="s3://pl-flash-data/optimized_tiny_imagenet",
        item_loader=InMemoryItemLoader(streaming_config=streaming_config),
        storage_options=storage_options
    )
    
    inmemory_times = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for i in range(min(test_samples, len(inmemory_dataset))):
            _ = inmemory_dataset[i]
        epoch_time = time.time() - epoch_start
        inmemory_times.append(epoch_time)
        print(f"  Epoch {epoch + 1}: {epoch_time:.2f}s")
        
        # Show memory stats after each epoch
        if hasattr(inmemory_dataset.item_loader, 'get_memory_stats'):
            stats = inmemory_dataset.item_loader.get_memory_stats()
            print(f"    💾 Memory: {stats['memory_utilization']:.1%} "
                  f"({stats['num_chunks_cached']} chunks)")
    
    inmemory_total = time.time() - start_time
    inmemory_avg = sum(inmemory_times) / len(inmemory_times)
    
    # Analysis
    print("\n📈 Performance Analysis")
    print("-" * 30)
    print(f"Standard Loader:")
    print(f"  Total time: {standard_total:.2f}s")
    print(f"  Average epoch: {standard_avg:.2f}s")
    print(f"  Consistency: {max(standard_times) / min(standard_times):.2f}x variation")
    
    print(f"\nInMemory Loader:")
    print(f"  Total time: {inmemory_total:.2f}s")
    print(f"  Average epoch: {inmemory_avg:.2f}s")
    print(f"  Consistency: {max(inmemory_times) / min(inmemory_times):.2f}x variation")
    
    # Calculate improvement (focusing on later epochs after caching)
    later_epochs_standard = sum(standard_times[2:]) / len(standard_times[2:])
    later_epochs_inmemory = sum(inmemory_times[2:]) / len(inmemory_times[2:])
    
    if later_epochs_inmemory < later_epochs_standard:
        speedup = later_epochs_standard / later_epochs_inmemory
        print(f"\n🚀 Later Epochs Speedup: {speedup:.2f}x")
    else:
        print(f"\n⏱️ Performance difference: {later_epochs_inmemory / later_epochs_standard:.2f}x")
    
    print(f"\n💡 Key Insight: Look for decreasing times in later epochs for in-memory loader")
    return inmemory_times, standard_times


def benchmark_random_access():
    """Test performance with random access patterns."""
    print("\n🎲 Benchmarking Random Access Performance")
    print("=" * 50)
    
    import random
    
    storage_options = {"config": Config(signature_version=UNSIGNED)}
    streaming_config = create_optimized_config(200)
    
    # Create datasets
    standard_dataset = StreamingDataset(
        input_dir="s3://pl-flash-data/optimized_tiny_imagenet",
        storage_options=storage_options
    )
    
    inmemory_dataset = StreamingDataset(
        input_dir="s3://pl-flash-data/optimized_tiny_imagenet",
        item_loader=InMemoryItemLoader(streaming_config=streaming_config),
        storage_options=storage_options
    )
    
    # Generate random access pattern
    dataset_size = min(100, len(standard_dataset))
    random_indices = [random.randint(0, dataset_size - 1) for _ in range(50)]
    
    # Warm up in-memory loader by accessing samples sequentially first
    print("🔥 Warming up in-memory cache...")
    for i in range(dataset_size):
        _ = inmemory_dataset[i]
    
    stats = inmemory_dataset.item_loader.get_memory_stats()
    print(f"   Cache warmed: {stats['memory_utilization']:.1%} memory used")
    
    # Test random access on standard loader
    print("\n📊 Testing random access - Standard Loader...")
    start_time = time.time()
    for idx in random_indices:
        _ = standard_dataset[idx]
    standard_random_time = time.time() - start_time
    
    # Test random access on in-memory loader
    print("🧠 Testing random access - InMemory Loader...")
    start_time = time.time()
    for idx in random_indices:
        _ = inmemory_dataset[idx]
    inmemory_random_time = time.time() - start_time
    
    print(f"\n📈 Random Access Results:")
    print(f"  Standard: {standard_random_time:.3f}s")
    print(f"  InMemory: {inmemory_random_time:.3f}s")
    
    if inmemory_random_time < standard_random_time:
        speedup = standard_random_time / inmemory_random_time
        print(f"  🎯 Random access speedup: {speedup:.2f}x")
    else:
        print(f"  ⏱️ Random access overhead: {inmemory_random_time / standard_random_time:.2f}x")


def benchmark_memory_efficiency():
    """Test memory management and eviction behavior."""
    print("\n🗃️ Benchmarking Memory Management")
    print("=" * 50)
    
    storage_options = {"config": Config(signature_version=UNSIGNED)}
    
    # Test with larger memory limit for better testing (was 10MB, now 500MB as requested)
    large_config = StreamingConfig(
        max_memory_size=500 * 1024 * 1024,  # 500MB for better testing
        eviction_policy=EvictionPolicy.LRU,
        chunk_timeout=30.0,
        enable_memory_monitoring=True,
    )
    
    loader = InMemoryItemLoader(streaming_config=large_config)
    dataset = StreamingDataset(
        input_dir="s3://pl-flash-data/optimized_tiny_imagenet",
        item_loader=loader,
        storage_options=storage_options
    )
    
    print(f"💾 Testing with {large_config.max_memory_size / (1024*1024):.0f}MB memory limit")
    
    # Access samples to trigger loading and monitor memory usage
    print("🔄 Loading samples into memory...")
    for i in range(min(50, len(dataset))):  # Test more samples to better show memory usage
        try:
            _ = dataset[i]
            
            if i % 10 == 0:
                stats = loader.get_memory_stats()
                print(f"  Sample {i:2d}: {stats['memory_utilization']:.1%} memory "
                      f"({stats['num_chunks_cached']} chunks cached)")
                
                # Show more detailed memory info if available
                if 'total_memory_usage' in stats:
                    memory_mb = stats['total_memory_usage'] / (1024 * 1024)
                    print(f"            {memory_mb:.1f}MB used")
                    
        except Exception as e:
            print(f"  ⚠️ Error accessing sample {i}: {e}")
            continue
    
    final_stats = loader.get_memory_stats()
    print(f"\n📊 Final Memory Stats:")
    print(f"  Memory utilization: {final_stats['memory_utilization']:.1%}")
    print(f"  Chunks cached: {final_stats['num_chunks_cached']}")
    print(f"  Total memory: {final_stats['total_memory_usage'] / (1024*1024):.1f}MB")
    
    # Test memory efficiency by accessing random samples
    print(f"\n🎯 Testing random access on cached data...")
    import random
    random_indices = [random.randint(0, min(49, len(dataset)-1)) for _ in range(10)]
    
    start_time = time.time()
    for idx in random_indices:
        try:
            _ = dataset[idx]
        except Exception as e:
            print(f"  ⚠️ Error accessing random sample {idx}: {e}")
    
    random_access_time = time.time() - start_time
    print(f"  ⏱️ Random access time: {random_access_time:.3f}s for 10 samples")
    print(f"  🚀 Average per sample: {random_access_time / 10 * 1000:.1f}ms")


def create_performance_report() -> Dict[str, Any]:
    """Generate comprehensive performance report."""
    print("📋 Generating Comprehensive Performance Report")
    print("=" * 50)
    
    report = {
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "s3://pl-flash-data/optimized_tiny_imagenet",
        "tests": {}
    }
    
    try:
        # Run repeated access benchmark
        inmemory_times, standard_times = benchmark_repeated_access()
        report["tests"]["repeated_access"] = {
            "inmemory_times": inmemory_times,
            "standard_times": standard_times,
            "improvement_pattern": [
                standard_times[i] / inmemory_times[i] for i in range(len(inmemory_times))
            ]
        }
        
        # Run random access benchmark  
        benchmark_random_access()
        
        # Run memory efficiency test
        benchmark_memory_efficiency()
        
        report["status"] = "completed"
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
        report["status"] = "failed"
        report["error"] = str(e)
    
    return report


def main():
    """Run enhanced performance tests."""
    print("🚀 Enhanced InMemoryItemLoader Performance Test")
    print("=" * 60)
    print("Testing optimized patterns for in-memory caching benefits...")
    print()
    
    try:
        report = create_performance_report()
        
        print("\n✅ Performance testing completed!")
        print("\n💡 Key Takeaways:")
        print("• In-memory caching shines with repeated access patterns")
        print("• First epoch may be slower due to streaming overhead")
        print("• Later epochs should show significant speedup")
        print("• Random access benefits most from warm cache")
        print("• Memory management automatically handles eviction")
        
        if report["status"] == "completed":
            print(f"\n📊 Report generated at: {report['test_timestamp']}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
