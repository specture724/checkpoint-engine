import concurrent.futures
from collections import defaultdict

from loguru import logger
from pydantic import BaseModel
import torch

from checkpoint_engine.types import BucketRange, H2DBucket, MemoryBuffer, MemoryBufferMetaList, ParameterMeta
from checkpoint_engine.io import _load_checkpoint

# 256 bytes alignment when flatten torch tensors to uint8 buffer
_ALIGN_SIZE = 256


def _align_size(dtype: torch.dtype, shape: torch.Size) -> int:
    return (dtype.itemsize * shape.numel() + _ALIGN_SIZE - 1) // _ALIGN_SIZE * _ALIGN_SIZE


def _to_named_tensor(metas: list[ParameterMeta], offset: int = 0) -> list[dict]:
    ret = []
    for meta in metas:
        size = _align_size(meta.dtype, meta.shape)
        ret.append(
            {
                "name": meta.name,
                "dtype": meta.dtype,
                "shape": meta.shape,
                "offset": offset,
            }
        )
        offset += size
    return ret

def _register_checkpoint(
    *,
    files: list[str],
    named_tensors: dict[str, torch.Tensor],
    rank: int | None = None,
) -> list[MemoryBuffer]:
    logger.info(
        f"[rank{rank}] start to register checkpoint with {len(files)} files and {len(named_tensors)} named_tensors"
    )
    if not files and not named_tensors:
        return []
    parameters = _load_checkpoint(files)
    if named_tensors:
        parameters.update(named_tensors)
    bucket_size = max(4 << 30, max(_align_size(x.dtype, x.shape) for x in parameters.values()))

    class MemoryBucket(BaseModel):
        size: int
        metas: list[ParameterMeta]

    buckets: list[MemoryBucket] = [MemoryBucket(size=0, metas=[])]
    for name, tensor in sorted(parameters.items()):
        size = _align_size(tensor.dtype, tensor.shape)
        if buckets[-1].size + size > bucket_size:
            assert buckets[-1], f"buckets[{len(buckets) - 1}] should not be empty"
            buckets.append(MemoryBucket(size=0, metas=[]))
        buckets[-1].metas.append(ParameterMeta(name=name, shape=tensor.shape, dtype=tensor.dtype))
        buckets[-1].size += size

    memory_buffers = [
        MemoryBuffer(buffer=torch.empty(0), size=bucket.size, metas=bucket.metas)
        for bucket in buckets
    ]

    def register_pin_memory(idx: int, size: int) -> tuple[int, torch.Tensor]:
        buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        return idx, buffer

    def register_tensor(buffer: torch.Tensor, offset: int, tensor: torch.Tensor):
        buffer[offset : offset + tensor.nbytes] = tensor.view(-1).view(dtype=torch.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(register_pin_memory, idx, bucket.size)
            for idx, bucket in enumerate(buckets)
        ]
        new_futures = []
        for future in concurrent.futures.as_completed(futures):
            idx, buffer = future.result()
            assert buffer.numel() == buckets[idx].size, (
                f"buffer numel {buffer.numel()} should be equal to bucket size {buckets[idx].size}"
            )
            memory_buffers[idx].buffer = buffer
            logger.info(
                f"[rank{rank}] register pin_memory for bucket {idx + 1}/{len(buckets)} finished, "
                f"size {buffer.numel() / 1024 / 1024:.2f}MiB, start to copy tensors to buffer"
            )
            offset = 0
            for meta in buckets[idx].metas:
                name = meta.name
                tensor = parameters[name]
                size = _align_size(tensor.dtype, tensor.shape)
                assert size == _align_size(meta.dtype, meta.shape), (
                    f"tensor {name} size {size} should be equal to meta size {_align_size(meta.dtype, meta.shape)}"
                )
                new_futures.append(executor.submit(register_tensor, buffer, offset, tensor))
                offset += size
        for future in concurrent.futures.as_completed(new_futures):
            future.result()
    return memory_buffers


def _gen_h2d_buckets(
    global_metas: dict[int, MemoryBufferMetaList],
    bucket_size: int,
    local_topo: dict[str, set[int]],
    remote_topo: dict[str, set[int]],
    ranks: list[int] | None = None,
) -> list[tuple[int, int, H2DBucket]]:
    buckets: list[tuple[int, H2DBucket]] = []

    for owner_rank, items in global_metas.items():
        buckets.append((owner_rank, H2DBucket(size=0, ranges=[], items=[])))
        for idx, metas in enumerate(items.memory_buffer_metas_list):
            start_offset, offset = 0, 0
            for meta in metas.metas:
                s = _align_size(meta.dtype, meta.shape)
                if buckets[-1][1].size + s > bucket_size:
                    if offset - start_offset > 0:
                        buckets[-1][1].ranges.append(
                            BucketRange(idx, start_offset, offset - start_offset)
                        )
                    start_offset = offset
                    buckets.append((owner_rank, H2DBucket(size=0, ranges=[], items=[])))
                offset += s
                buckets[-1][1].size += s
                buckets[-1][1].items.append(meta)
            buckets[-1][1].ranges.append(BucketRange(idx, start_offset, offset - start_offset))
        assert buckets[-1][1].size > 0, (
            f"buckets[-1][1].size {buckets[-1][1].size} should be greater than 0"
        )
    ranks_set = set(ranks) if ranks else set()
    actual_local_topo = (
        {k: v & ranks_set for k, v in local_topo.items() if v & ranks_set} if ranks else local_topo
    )
    # if ranks is empty, assign the owner_rank as receiver_rank, this is used for colocate architecture
    if not ranks:
        return [(owner_rank, owner_rank, bucket) for owner_rank, bucket in buckets]
    else:
        return _assign_receiver_ranks(buckets, actual_local_topo, remote_topo)


def _assign_receiver_ranks(
    buckets: list[tuple[int, "T"]],
    local_topo: dict[str, set[int]],
    remote_topo: dict[str, set[int]],
) -> list[tuple[int, int, "T"]]:
    """
    (owner_rank, bucket) -> (receiver_rank, owner_rank, bucket)

    Assign receiver ranks to buckets. If ranks is empty, assign the owner_rank as receiver_rank.
    GPU-rdma_device topology will be considered to make full use of the bandwidth.
    """
    if not buckets:
        logger.warning("bucket list is empty, no need to assign receiver ranks")
        return []
    rank_to_rdma_device = {
        rank: rdma_device for rdma_device, ranks in remote_topo.items() for rank in ranks
    }

    # group buckets by owner RDMA devices
    buckets_by_rdma_device = defaultdict(list)
    for owner_rank, bucket in buckets:
        owner_rdma_device = rank_to_rdma_device[owner_rank]
        buckets_by_rdma_device[owner_rdma_device].append((owner_rank, bucket))

    buckets_matrix = list(buckets_by_rdma_device.values())
    assert buckets_matrix, "buckets_matrix should not be empty"

    # Select receiver ranks. We use the minimum rank in each local RDMA device group as receiver rank
    num_receivers = min(len(local_topo), len(buckets_by_rdma_device))
    receiver_list = [min(ranks) for ranks in list(local_topo.values())[:num_receivers]]

    flattened_buckets = [
        buckets_matrix[row][col]
        for col in range(
            max(len(matrix_row) for matrix_row in buckets_matrix) if buckets_matrix else 0
        )
        for row in range(len(buckets_matrix))
        if col < len(buckets_matrix[row])
    ]

    buckets_with_receiver = []
    assigned_cnt = 0
    while assigned_cnt < len(flattened_buckets):
        occupied_devices = set()
        for receiver_rank in receiver_list:
            if assigned_cnt >= len(flattened_buckets):
                break
            owner_rank, bucket = flattened_buckets[assigned_cnt]
            rdma_device = rank_to_rdma_device[owner_rank]
            if rdma_device in occupied_devices:
                break
            buckets_with_receiver.append((receiver_rank, owner_rank, bucket))
            occupied_devices.add(rdma_device)
            assigned_cnt += 1

    return buckets_with_receiver

