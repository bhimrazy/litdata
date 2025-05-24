#!/usr/bin/env python3
"""
Test script to validate the InMemoryItemLoader with actual optimized S3 data.

This script tests our in-memory item loader implementation against real optimized data
from S3 (specifically the pl-flash-data/optimized_tiny_imagenet dataset) to ensure
compatibility and performance with cloud storage.
"""

import sys
import time
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from litdata import StreamingDataset
from litdata.streaming.in_memory_item_loader import InMemoryItemLoader
from litdata.streaming.streaming_buffer import EvictionPolicy, StreamingConfig


def test_basic_s3_compatibility():
    """Test basic S3 compatibility with unsigned requests."""
    print("🔍 Testing basic S3 compatibility...")

    try:
        # Configure boto3 for unsigned requests
        s3_config = Config(signature_version=UNSIGNED)
        storage_options = {"config": s3_config}

        dataset = StreamingDataset(
            input_dir="s3://pl-flash-data/optimized_tiny_imagenet",
            storage_options=storage_options
        )

        print("   ✅ Dataset created successfully")
        print(f"   📊 Dataset length: {len(dataset)}")

        # Test basic data access
        sample = dataset[0]
        print("   ✅ Sample accessed successfully")
        print(f"   📝 Sample type: {type(sample)}")
        if hasattr(sample, "keys"):
            print(f"   🔑 Sample keys: {list(sample.keys())}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   📍 Traceback: {traceback.format_exc()}")
        return False


def test_in_memory_loader_with_s3():
    """Test the InMemoryItemLoader with real S3 data."""
    print("\n🧪 Testing InMemoryItemLoader with S3 data...")

    try:
        # Configure boto3 for unsigned requests
        s3_config = Config(signature_version=UNSIGNED)
        storage_options = {"config": s3_config}

        # Configure streaming for in-memory loading
        streaming_config = StreamingConfig(
            max_memory_size=100 * 1024 * 1024,  # 100MB
            eviction_policy=EvictionPolicy.LRU,
            chunk_timeout=30.0,
            enable_memory_monitoring=True,
        )

        # Create in-memory item loader
        loader = InMemoryItemLoader(streaming_config=streaming_config)

        # Test with S3 dataset using our in-memory loader
        dataset = StreamingDataset(
            input_dir="s3://pl-flash-data/optimized_tiny_imagenet", 
            item_loader=loader,
            storage_options=storage_options
        )

        print(f"   ✅ InMemory dataset created successfully")
        print(f"   📊 Dataset length: {len(dataset)}")

        # Test memory stats
        memory_stats = loader.get_memory_stats()
        print(f"   💾 Initial memory stats: {memory_stats}")

        # Test data access and streaming
        start_time = time.time()
        samples_tested = 0
        max_samples = min(10, len(dataset))  # Test first 10 samples or all if less

        for i in range(max_samples):
            sample = dataset[i]
            samples_tested += 1

            if i == 0:
                print(f"   📝 First sample type: {type(sample)}")
                if hasattr(sample, "keys"):
                    print(f"   🔑 First sample keys: {list(sample.keys())}")

            if (i + 1) % 5 == 0:
                stats = loader.get_memory_stats()
                print(f"   📈 After {i + 1} samples - Memory stats: {stats}")

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"   ✅ Successfully accessed {samples_tested} samples")
        print(f"   ⏱️ Time elapsed: {elapsed:.2f}s")
        print(f"   🚀 Samples per second: {samples_tested / elapsed:.2f}")

        # Final memory stats
        final_stats = loader.get_memory_stats()
        print(f"   💾 Final memory stats: {final_stats}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   📍 Traceback: {traceback.format_exc()}")
        return False


def test_memory_eviction_with_s3():
    """Test memory eviction behavior with S3 data."""
    print("\n🗃️ Testing memory eviction with S3 data...")

    try:
        # Configure boto3 for unsigned requests
        s3_config = Config(signature_version=UNSIGNED)
        storage_options = {"config": s3_config}

        # Configure very small memory limit to trigger eviction
        streaming_config = StreamingConfig(
            max_memory_size=50 * 1024,  # 50KB - very small to force eviction
            eviction_policy=EvictionPolicy.LRU,
            chunk_timeout=30.0,
            enable_memory_monitoring=True,
        )

        loader = InMemoryItemLoader(streaming_config=streaming_config)

        dataset = StreamingDataset(
            input_dir="s3://pl-flash-data/optimized_tiny_imagenet", 
            item_loader=loader,
            storage_options=storage_options
        )

        print("   ✅ Small memory dataset created")
        print(f"   💾 Memory limit: {streaming_config.max_memory_size} bytes")

        # Access several samples to trigger eviction
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            stats = loader.get_memory_stats()
            utilization = stats.get("memory_utilization", 0)
            print(f"   📊 Sample {i}: Memory utilization {utilization:.1%}")

            if utilization > 0.8:  # High memory usage
                print("   🚨 High memory usage detected, eviction likely occurring")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   📍 Traceback: {traceback.format_exc()}")
        return False


def test_performance_comparison():
    """Compare performance between standard and in-memory loaders."""
    print("\n🏎️ Testing performance comparison...")

    try:
        # Configure boto3 for unsigned requests
        s3_config = Config(signature_version=UNSIGNED)
        storage_options = {"config": s3_config}
        
        test_samples = 20  # More samples for better comparison
        repeat_accesses = 3  # Test multiple accesses to same data

        # Test standard loader with repeated access
        print("   📊 Testing standard loader with repeated access...")
        start_time = time.time()

        standard_dataset = StreamingDataset(
            input_dir="s3://pl-flash-data/optimized_tiny_imagenet",
            storage_options=storage_options
        )

        # Access samples multiple times (simulating epoch-based training)
        for repeat in range(repeat_accesses):
            for i in range(min(test_samples, len(standard_dataset))):
                _ = standard_dataset[i]

        standard_time = time.time() - start_time
        print(f"   ⏱️ Standard loader time ({repeat_accesses} repeats): {standard_time:.2f}s")

        # Test in-memory loader with repeated access
        print("   🧠 Testing in-memory loader with repeated access...")
        start_time = time.time()

        streaming_config = StreamingConfig(
            max_memory_size=200 * 1024 * 1024,  # 200MB
            eviction_policy=EvictionPolicy.LRU,
            chunk_timeout=30.0,
            enable_memory_monitoring=True,
        )

        inmemory_dataset = StreamingDataset(
            input_dir="s3://pl-flash-data/optimized_tiny_imagenet",
            item_loader=InMemoryItemLoader(streaming_config=streaming_config),
            storage_options=storage_options
        )

        # Access samples multiple times (first time loads, subsequent are from cache)
        for repeat in range(repeat_accesses):
            for i in range(min(test_samples, len(inmemory_dataset))):
                _ = inmemory_dataset[i]

        inmemory_time = time.time() - start_time
        print(f"   ⏱️ In-memory loader time ({repeat_accesses} repeats): {inmemory_time:.2f}s")

        # Calculate speedup
        if inmemory_time < standard_time:
            speedup = standard_time / inmemory_time
            print(f"   🚀 In-memory loader is {speedup:.2f}x faster!")
        else:
            slowdown = inmemory_time / standard_time
            print(f"   🐌 In-memory loader is {slowdown:.2f}x slower (expected for single pass)")

        # Test random access performance
        print("   🎲 Testing random access performance...")
        import random
        random_indices = [random.randint(0, min(test_samples-1, len(standard_dataset)-1)) for _ in range(10)]

        # Standard loader random access
        start_time = time.time()
        for idx in random_indices:
            _ = standard_dataset[idx]
        standard_random_time = time.time() - start_time

        # In-memory loader random access (should be faster due to caching)
        start_time = time.time()
        for idx in random_indices:
            _ = inmemory_dataset[idx]
        inmemory_random_time = time.time() - start_time

        print(f"   ⏱️ Standard random access: {standard_random_time:.2f}s")
        print(f"   ⏱️ In-memory random access: {inmemory_random_time:.2f}s")
        
        if inmemory_random_time < standard_random_time:
            random_speedup = standard_random_time / inmemory_random_time
            print(f"   🎯 Random access speedup: {random_speedup:.2f}x")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   📍 Traceback: {traceback.format_exc()}")
        return False


def test_streaming_downloader_directly():
    """Test streaming downloader components directly."""
    print("\n📡 Testing streaming downloader directly...")

    try:
        from litdata.streaming.streaming_buffer import StreamingChunkBuffer
        from litdata.streaming.streaming_downloader import StreamingDownloaderManager

        # Initialize components
        streaming_config = StreamingConfig(
            max_memory_size=50 * 1024 * 1024,  # 50MB
            eviction_policy=EvictionPolicy.LRU,
            chunk_timeout=30.0,
            enable_memory_monitoring=True,
        )

        buffer = StreamingChunkBuffer(streaming_config)
        manager = StreamingDownloaderManager()

        print("   ✅ Streaming components initialized")

        # Test S3 URL support
        s3_url = "s3://pl-flash-data/optimized_tiny_imagenet/chunk-0-0.bin"
        downloader = manager.get_downloader(s3_url)

        if downloader:
            print(f"   ✅ S3 downloader found: {type(downloader).__name__}")
            print(f"   🔗 Supports URL: {downloader.supports_url(s3_url)}")
        else:
            print(f"   ❌ No downloader found for S3 URL")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   📍 Traceback: {traceback.format_exc()}")
        return False


def run_all_tests():
    """Run all test scenarios."""
    print("🧪 In-Memory Item Loader S3 Test Suite")
    print("=" * 50)

    tests = [
        ("Basic S3 Compatibility", test_basic_s3_compatibility),
        ("InMemory Loader with S3", test_in_memory_loader_with_s3),
        ("Memory Eviction", test_memory_eviction_with_s3),
        ("Performance Comparison", test_performance_comparison),
        ("Streaming Downloader", test_streaming_downloader_directly),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🏃 Running: {test_name}")
        print("-" * 30)
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            results[test_name] = False
            print(f"   ❌ FAILED with exception: {e}")

    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {test_name}")

    print(f"\n🎯 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! In-memory loader is ready for S3!")
    else:
        print("⚠️ Some tests failed. Review the errors above.")

    return passed == total


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print(f"📍 Traceback: {traceback.format_exc()}")
        exit(1)
