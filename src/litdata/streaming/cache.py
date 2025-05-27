# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import psutil
from typing import Any, Dict, List, Optional, Tuple, Union

from litdata.constants import (
    _INDEX_FILENAME,
)
from litdata.streaming.item_loader import BaseItemLoader, Interval
from litdata.streaming.reader import BinaryReader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer
from litdata.streaming.writer import BinaryWriter
from litdata.utilities.encryption import Encryption
from litdata.utilities.env import _DistributedEnv, _WorkerEnv
from litdata.utilities.format import _convert_bytes_to_int

logger = logging.Logger(__name__)


def _get_optimal_cache_size(requested_size: Union[int, str] = "100GB") -> int:
    """Calculate optimal cache size based on available memory and performance guidelines."""
    # Convert requested size to int
    requested_bytes = _convert_bytes_to_int(requested_size) if isinstance(requested_size, str) else requested_size
    
    # Get available memory
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)
    
    # Performance thresholds
    MIN_RECOMMENDED_GB = 25
    OPTIMAL_RATIO = 0.3  # Use max 30% of available memory for cache
    
    # Calculate optimal size
    optimal_bytes = min(
        requested_bytes,
        int(memory_info.available * OPTIMAL_RATIO)
    )
    
    optimal_gb = optimal_bytes / (1024**3)
    requested_gb = requested_bytes / (1024**3)
    
    # Issue warnings for suboptimal configurations
    if optimal_gb < MIN_RECOMMENDED_GB:
        if available_gb < MIN_RECOMMENDED_GB:
            logger.warning(
                f"Available memory ({available_gb:.1f}GB) is less than recommended minimum "
                f"({MIN_RECOMMENDED_GB}GB) for optimal performance. Consider using a machine "
                f"with more memory or reducing batch sizes."
            )
        else:
            logger.warning(
                f"Cache size ({optimal_gb:.1f}GB) is below recommended minimum "
                f"({MIN_RECOMMENDED_GB}GB). Performance may be suboptimal. "
                f"Available memory: {available_gb:.1f}GB"
            )
    
    if requested_gb > available_gb * 0.5:
        logger.warning(
            f"Requested cache size ({requested_gb:.1f}GB) exceeds 50% of available memory "
            f"({available_gb:.1f}GB). This may cause memory pressure and swapping."
        )
    
    return optimal_bytes


def _get_optimal_pre_download(cache_size_bytes: int, chunk_avg_size: int = 64 * 1024 * 1024) -> int:
    """Calculate optimal pre-download count based on cache size and chunk characteristics."""
    # Base recommendation: 2-10 chunks based on cache size
    cache_gb = cache_size_bytes / (1024**3)
    
    if cache_gb >= 50:
        return 8  # High-performance setup
    if cache_gb >= 25:
        return 5  # Standard setup
    if cache_gb >= 10:
        return 3  # Limited memory
    return 2  # Minimal setup


class Cache:
    def __init__(
        self,
        input_dir: Optional[Union[str, Dir]],
        subsampled_files: Optional[List[str]] = None,
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
        compression: Optional[str] = None,
        encryption: Optional[Encryption] = None,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[Union[int, str]] = None,
        item_loader: Optional[BaseItemLoader] = None,
        max_cache_size: Union[int, str] = "100GB",
        serializers: Optional[Dict[str, Serializer]] = None,
        writer_chunk_index: Optional[int] = None,
        storage_options: Optional[Dict] = {},
        session_options: Optional[Dict] = {},
        max_pre_download: int = 2,
    ):
        """The Cache enables to optimise dataset format for cloud training. This is done by grouping several elements
        together in order to accelerate fetching.

        Args:
            input_dir: The path to where the chunks will be or are stored.
            subsampled_files: List of subsampled chunk files loaded from `input_dir/index.json` file.
            region_of_interest: List of tuples of (start,end) of region of interest for each chunk.
            compression: The name of the algorithm to reduce the size of the chunks.
            encryption: The encryption algorithm to use.
            chunk_bytes: The maximum number of bytes within a chunk.
            chunk_size: The maximum number of items within a chunk.
            item_loader: The object responsible to generate the chunk intervals and load an item froma chunk.
            max_cache_size: The maximum cache size used by the reader when fetching the chunks.
            serializers: Provide your own serializers.
            writer_chunk_index: The index of the chunk to start from when writing.
            storage_options: Additional connection options for accessing storage services.
            session_options: Additional options for the S3 session.
            max_pre_download: Maximum number of chunks that can be pre-downloaded while filling up the cache.

        """
        super().__init__()
        input_dir = _resolve_dir(input_dir)
        self._cache_dir = input_dir.path
        assert self._cache_dir
        self._writer = BinaryWriter(
            self._cache_dir,
            chunk_size=chunk_size,
            chunk_bytes=chunk_bytes,
            compression=compression,
            encryption=encryption,
            serializers=serializers,
            chunk_index=writer_chunk_index or 0,
            item_loader=item_loader,
        )
        # Optimize cache configuration using system resources
        optimized_cache_size = _get_optimal_cache_size(max_cache_size)
        optimized_pre_download = _get_optimal_pre_download(optimized_cache_size)
        
        # Use optimized values, but allow user override for max_pre_download
        final_pre_download = max_pre_download if max_pre_download != 2 else optimized_pre_download
        
        logger.debug(
            f"Cache configuration: size={optimized_cache_size / (1024**3):.1f}GB, "
            f"pre_download={final_pre_download}"
        )
        
        self._reader = BinaryReader(
            self._cache_dir,
            subsampled_files=subsampled_files,
            region_of_interest=region_of_interest,
            max_cache_size=optimized_cache_size,
            remote_input_dir=input_dir.url,
            compression=compression,
            encryption=encryption,
            item_loader=item_loader,
            serializers=serializers,
            storage_options=storage_options,
            session_options=session_options,
            max_pre_download=final_pre_download,
        )
        self._is_done = False
        self._distributed_env = _DistributedEnv.detect()
        self._rank: Optional[int] = None

    @property
    def rank(self) -> int:
        """Returns the rank of the Cache."""
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._distributed_env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    @property
    def filled(self) -> bool:
        """Returns whether the caching phase is done."""
        if self._is_done:
            return True
        assert self._cache_dir
        self._is_done = os.path.exists(os.path.join(self._cache_dir, _INDEX_FILENAME))
        return self._is_done

    @property
    def cache_dir(self) -> str:
        assert self._cache_dir
        return self._cache_dir

    @property
    def checkpoint_dir(self) -> str:
        assert self._cache_dir
        checkpoint_dir = os.path.join(self._cache_dir, "checkpoints")
        return self._try_create(checkpoint_dir)

    def _try_create(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def __setitem__(self, index: int, data: Any) -> None:
        """Store an item in the writer."""
        self._writer[index] = data

    def _add_item(self, index: int, data: Any) -> Optional[str]:
        """Store an item in the writer and optionally return the chunk path."""
        return self._writer.add_item(index, data)

    def __getitem__(self, index: Union[int, ChunkedIndex]) -> Dict[str, Any]:
        """Read an item in the reader."""
        if isinstance(index, int):
            index = ChunkedIndex(*self._get_chunk_index_from_index(index))
        return self._reader.read(index)

    def done(self) -> Optional[List[str]]:
        """Inform the writer the chunking phase is finished."""
        return self._writer.done()

    def merge(self, num_workers: int = 1, node_rank: Optional[int] = None) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer.merge(num_workers, node_rank=node_rank)

    def _merge_no_wait(self, node_rank: Optional[int] = None, existing_index: Optional[Dict[str, Any]] = None) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer._merge_no_wait(node_rank=node_rank, existing_index=existing_index)

    def __len__(self) -> int:
        return self._reader.get_length()

    def get_chunk_intervals(self) -> List[Interval]:
        return self._reader.get_chunk_intervals()

    def _get_chunk_index_from_index(self, index: int) -> Tuple[int, int]:
        return self._reader._get_chunk_index_from_index(index)

    def save_checkpoint(self, checkpoint_dir: str = ".checkpoints") -> Optional[str]:
        """Save the current state of the writer to a checkpoint."""
        return self._writer.save_checkpoint(checkpoint_dir=checkpoint_dir)
