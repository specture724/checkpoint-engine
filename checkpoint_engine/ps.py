import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import timedelta

import torch
import torch.distributed as dist
import zmq
from loguru import logger
from torch.multiprocessing.reductions import reduce_tensor

from checkpoint_engine.device_utils import DeviceManager, get_ip
from checkpoint_engine.memory_layout import (
    _ALIGN_SIZE,
    _align_size,
    _gen_h2d_buckets,
    _register_checkpoint,
    _to_named_tensor,
)
from checkpoint_engine.p2p_store import P2PStore, _get_master_port, _get_physical_gpu_id
from checkpoint_engine.types import (
    DataToGather,
    H2DBucket,
    MemoryBuffer,
    MemoryBufferMetaList,
    MemoryBufferMetas,
)


def _get_bcast_rank_map(world_size: int, ranks: list[int] | None) -> dict[int, int]:
    """
    map the real ranks (receiver_rank) to the bcast ranks (0 ~ len(ranks) - 1),
    which are generated in self.init_process_group_for_ranks
    """
    bcast_rank_map: dict[int, int] = {}
    if not ranks:
        bcast_rank_map = {r: r for r in range(world_size)}
    else:
        for i, r in enumerate(ranks):
            bcast_rank_map[r] = i
    return bcast_rank_map


class ParameterServer:
    def __init__(
        self,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        auto_pg: bool = False,
        gpu_count: int | None = None,
        mem_fraction: float | None = None,
    ):
        """
        Initialize the parameter server. env RANK, WORLD_SIZE and MASTER_ADDR must be set.

        Args:
            auto_pg: Whether to automatically initialize the process group.
                Notice that if auto_pg is True, will destroy the process group after update.
            mem_fraction: The proportion (as a fraction) of the current free CUDA memory for allocation.
        """
        self._rank = rank or int(os.environ.get("RANK", None))
        self._world_size = world_size or int(os.environ.get("WORLD_SIZE", None))
        self.device_manager = DeviceManager()
        self._gpu_count = gpu_count or self.device_manager.device_module.device_count()
        self._local_rank = self._rank % self._gpu_count
        self._auto_pg = auto_pg
        self._all_hosts = []
        self._global_device_uuids: list[str] = []
        self._local_rdma_devices: dict[str, set[int]] = defaultdict(set)
        self._remote_rdma_devices: dict[str, set[int]] = defaultdict(set)
        self._mem_fraction = mem_fraction or 0.9

        assert self._rank is not None and self._rank >= 0, self._rank
        assert self._world_size and self._world_size > 0, self._world_size
        assert (
            self._gpu_count is not None
            and self._gpu_count > 0
            and self._gpu_count <= self.device_manager.device_module.device_count()
        ), self._gpu_count
        assert (
            self._mem_fraction is not None and self._mem_fraction > 0 and self._mem_fraction <= 1
        ), self._mem_fraction

        self._zmq_ctx = zmq.Context()
        self._zmq_addr_counter = 0

        self._memory_pool: dict[str, list[MemoryBuffer]] = {}
        # dict key is owner_rank, value is a bucket metas list in owner_rank
        self._current_global_parameter_metas: dict[int, MemoryBufferMetaList] = {}
        try:
            self._p2p_store = P2PStore(self.device_manager)
        except ImportError as e:
            logger.warning(f"[rank{self._rank}] fail to initialize p2p store due to {e}")
            self._p2p_store = None

        device_index = self._local_rank
        self.device_manager.device_module.set_device(device_index)
        self._device_uuid = _get_physical_gpu_id(self.device_manager, device_index)
        self._rdma_device = None if self._p2p_store is None else self._p2p_store.device

    def _logger_rank0(self, msg: str):
        if self._local_rank == 0:
            logger.info(msg)

    def get_metas(self) -> dict[int, MemoryBufferMetaList]:
        return self._current_global_parameter_metas

    def load_metas(self, metas: dict[int, MemoryBufferMetaList]):
        self._current_global_parameter_metas = metas
        self._remote_rdma_devices = defaultdict(set)
        for i, meta in self._current_global_parameter_metas.items():
            assert meta.rdma_device is not None, "meta.rdma_device should not be None"
            assert meta.p2p_store_addr is not None, "meta.p2p_store_addr should not be None"
            self._remote_rdma_devices[
                meta.rdma_device + "@" + meta.p2p_store_addr.split(":")[0]
            ].add(i)

    def register_checkpoint(
        self,
        checkpoint_name: str,
        *,
        files: list[str] | None = None,
        named_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Register a checkpoint to the parameter server. Both files and named_tensors will be registered together.

        Args:
            checkpoint_name: The name of the checkpoint.
            files: The safetensors files to register.
            named_tensors: The named tensors to register.
        """
        try:
            assert checkpoint_name not in self._memory_pool, (
                f"checkpoint {checkpoint_name} already registered"
            )
            self._memory_pool[checkpoint_name] = _register_checkpoint(
                files=files or [], named_tensors=named_tensors or {}, rank=self._rank
            )
            if self._p2p_store is not None:
                self._register_parameters_to_p2p_store(checkpoint_name)
        except Exception:
            logger.exception(
                f"[rank{self._rank}] fail to register checkpoint {checkpoint_name} with files {files}"
            )
            if self._p2p_store is not None:
                self._unregister_parameters_from_p2p_store(checkpoint_name)
            self.unregister_checkpoint(checkpoint_name)
            raise

    def unregister_checkpoint(self, checkpoint_name: str):
        """
        Unregister a checkpoint from the parameter server. This function will also unregister the checkpoint
        from p2p store if p2p store is initialized.
        """
        if checkpoint_name not in self._memory_pool:
            return
        if self._p2p_store is not None:
            num_unregistered = self._unregister_parameters_from_p2p_store(checkpoint_name)
            logger.info(
                f"[rank{self._rank}] unregister {num_unregistered} parameters from p2p store for checkpoint {checkpoint_name}"
            )
        del self._memory_pool[checkpoint_name]
        # see https://github.com/pytorch/pytorch/blob/31d5c675394705f8a6bc767f80ae14bf4f01246b/torch/csrc/cuda/Module.cpp#L2018
        # this works by using torch>=2.5.0
        torch._C._host_emptyCache()

    def gather_metas(self, checkpoint_name: str):
        """
        Gather the parameter metas from all ranks. This will gather memory_buffer, and other metadatas.
        This function should be called before update and init a new value to `self._current_global_parameter_metas`,
        which can be exported by using `self.get_metas` function.
        """
        if self._auto_pg and not dist.is_initialized():
            self.init_process_group()
        assert dist.is_initialized(), "process group is not initialized"
        metas_lst: list[DataToGather | None] = [None for _ in range(self._world_size)]  # type: ignore
        metas = DataToGather(
            memory_buffer_metas_list=[
                MemoryBufferMetas(
                    metas=x.metas,
                    ptr=x.buffer.data_ptr(),
                    size=x.size,
                )
                for x in self._memory_pool.get(checkpoint_name, [])
            ],
            p2p_store_addr=None if self._p2p_store is None else self._p2p_store.addr,
            host_ip=get_ip(),
            device_uuid=self._device_uuid,
            rdma_device=self._rdma_device or "",
        )

        dist.all_gather_object(metas_lst, metas)

        num_parameters = 0
        all_hosts: list[str] = []
        global_device_uuids: list[str] = []
        for i, metas_buckets in enumerate(metas_lst):
            assert metas_buckets is not None, f"metas_buckets {i} should not be None"
            if i % self._gpu_count == 0 and not self._all_hosts:
                all_hosts.append(metas_buckets.host_ip)
            if not self._global_device_uuids:
                global_device_uuids.append(metas_buckets.device_uuid)
            if metas_buckets.memory_buffer_metas_list:
                self._current_global_parameter_metas[i] = MemoryBufferMetaList(
                    memory_buffer_metas_list=metas_buckets.memory_buffer_metas_list,
                    p2p_store_addr=metas_buckets.p2p_store_addr,
                    rdma_device=metas_buckets.rdma_device,
                )
                num_parameters += sum(len(x.metas) for x in metas_buckets.memory_buffer_metas_list)
            self._local_rdma_devices[
                metas_buckets.rdma_device + "@" + metas_buckets.p2p_store_addr.split(":")[0]
                if metas_buckets.p2p_store_addr
                else metas_buckets.host_ip
            ].add(i)
        if not self._all_hosts:
            self._all_hosts = all_hosts
        if not self._global_device_uuids:
            self._global_device_uuids = global_device_uuids
        # Sender node and Receiver node have the same GPU-rdma_device topology is considered as default.
        # Rewrite the sender's topology (_remote_rdma_devices) by calling load_metas.
        self._remote_rdma_devices = self._local_rdma_devices.copy()
        logger.info(
            f"[rank{self._rank}] gather parameter metas finished, num_parameters: {num_parameters}"
        )

    def init_process_group(
        self,
        *,
        master_addr: str | None = None,
        master_port: int | None = None,
        timeout: timedelta = timedelta(minutes=10),
    ):
        """
        Initialize the process group for the ranks. This global group can be easily destroyed by calling dist.destroy_process_group.

        Args:
            master_port: The specified port of the master node. If not set, will use _get_master_port to get the port.
            timeout: The timeout of the process group.
        """
        master_addr = master_addr or os.getenv("MASTER_ADDR")
        assert master_addr, "master_addr is required"
        store = dist.TCPStore(
            master_addr,
            _get_master_port(master_port),
            self._world_size,
            timeout=timeout,
            is_master=self._rank == 0,
        )
        dist.init_process_group(
            backend=self.device_manager.backend,
            world_size=self._world_size,
            rank=self._rank,
            timeout=timeout,
            store=store,
        )
        logger.info(f"[rank{self._rank}] init process group successfully.")

    def update(
        self,
        checkpoint_name: str,
        req_func: Callable[[list[tuple[str, str]]], None],
        *,
        ranks: list[int] | None = None,
    ) -> None:
        """
        Update the checkpoint to inference engine. This function should be called after gather_metas.

        Args:
            checkpoint_name: The name of the checkpoint.
            req_func: The function to request the inference of inference engine.
            ranks: The ranks to update. If not set, will use fully broadcast to update to all ranks,
                which is the fastest way to update weights, especially in colocated architecture.
                If set, will use p2p to update to the ranks, this is flexible to update to a group of ranks,
                which is useful in disaggregated architecture.
        """
        assert req_func is not None, "req_func is required"
        try:
            # if both ranks is None or [], it will use fully broadcast to update to all ranks
            if not ranks:
                if self._auto_pg and not dist.is_initialized():
                    self.init_process_group()
                self._update_per_bucket(checkpoint_name, req_func)
            else:
                if self._auto_pg:
                    if dist.is_initialized():
                        dist.destroy_process_group()
                        # HACK: wait 2s to ensure destroy is finished
                        time.sleep(2)
                    self.init_process_group_for_ranks(ranks)
                if self._rank not in ranks:
                    return
                self._update_per_bucket(checkpoint_name, req_func, ranks)
            if self._auto_pg:
                dist.destroy_process_group()

            self.device_manager.device_module.empty_cache()

            logger.info(
                f"[rank{self._rank}] update checkpoint {checkpoint_name} with ranks {ranks} done. "
                f"Current CUDA allocated {self.device_manager.device_module.memory_allocated() / 1024 / 1024} MB, "
                f"reserved {self.device_manager.device_module.memory_reserved() / 1024 / 1024} MB."
            )
        except Exception as e:
            logger.exception(
                f"[rank{self._rank}] update checkpoint {checkpoint_name} with ranks {ranks} error {e}"
            )
            raise

    def _bind_zmq_socket(self) -> tuple[zmq.Socket, list[tuple[str, str]]]:
        def zmq_handle(device_uuid: str) -> str:
            return f"ipc://@checkpoint-engine-{device_uuid}-{self._zmq_addr_counter}.sock"

        socket_paths = [(uid, zmq_handle(uid)) for uid in self._global_device_uuids]
        socket = self._zmq_ctx.socket(zmq.REQ)
        socket.bind(zmq_handle(self._device_uuid))
        self._zmq_addr_counter += 1
        return socket, socket_paths

    def _detect_bucket_size(self, *, disable_h2d_buffer: bool = False) -> tuple[int, bool]:
        GiB = 1 << 30  # noqa: N806
        # auto detect bucket size
        tensor = torch.tensor(
            [
                # proportion of current cuda free memory bytes
                int(
                    float(self.device_manager.device_module.mem_get_info()[0]) * self._mem_fraction
                ),
                # we use negative value to reuse allreduce min operation
                # for getting the max value of zmq_addr_counter in all ranks
                -self._zmq_addr_counter,
            ],
            dtype=torch.int64,
            device=self.device_manager.device_type,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        tensor = tensor.cpu()
        free_bytes, self._zmq_addr_counter = tensor[0].item(), -tensor[1].item()
        max_tensor_bytes = 0
        for items in self._current_global_parameter_metas.values():
            for metas_list in items.memory_buffer_metas_list:
                for meta in metas_list.metas:
                    max_tensor_bytes = max(max_tensor_bytes, _align_size(meta.dtype, meta.shape))
        free_bytes_divided_3 = free_bytes // (3 * _ALIGN_SIZE) * _ALIGN_SIZE
        if max_tensor_bytes <= free_bytes_divided_3 and not disable_h2d_buffer:
            self._logger_rank0(f"[rank{self._rank}] use h2d buffer")
            # using h2d_buffer can make all ranks' h2d parallel execution
            # the cost is that we need to allocate extra h2d_buffer's GPU memory
            free_bytes = free_bytes_divided_3
        else:
            # if the memory is not enough, it will fallback to disable_h2d_buffer mode,
            # at this time, the bandwidth will be limited by the h2d of a single machine,
            # but we can save GPU memory
            self._logger_rank0(
                f"[rank{self._rank}] disable h2d buffer when max_tensor_bytes {max_tensor_bytes} is larger than free_bytes {free_bytes} // 3"
            )
            free_bytes = free_bytes // (2 * _ALIGN_SIZE) * _ALIGN_SIZE
            assert max_tensor_bytes <= free_bytes, (
                f"max_tensor_bytes {max_tensor_bytes} should be less than free_bytes {free_bytes}"
            )
            disable_h2d_buffer = True
        max_bytes = int(os.getenv("PS_MAX_BUCKET_SIZE_GB", 8)) * GiB
        bucket_size = min(max(max_bytes, max_tensor_bytes), free_bytes)
        logger.info(f"[rank{self._rank}] auto detect bucket size {bucket_size / GiB:.2f} GiB")
        return bucket_size, disable_h2d_buffer

    def _copy_to_buffer(
        self,
        checkpoint_name: str,
        bucket: H2DBucket,
        buffer: torch.Tensor,
        owner_rank: int | None = None,
    ):
        offset = 0
        if owner_rank is not None:
            buf_ptrs, remote_ptrs, lens = [], [], []
            ptr_base = buffer.data_ptr()
            target_addr, ptrs = self._get_addr_ptrs(owner_rank)
        for b in bucket.ranges:
            assert offset + b.size <= bucket.size, (
                f"offset {offset} + size {b.size} > bucket_size {bucket.size}"
            )
            if owner_rank is not None:
                buf_ptrs.append(ptr_base + offset)
                remote_ptrs.append(ptrs[b.idx][0] + b.offset)
                lens.append(b.size)
            else:
                pool = self._memory_pool[checkpoint_name][b.idx]
                buffer[offset : offset + b.size].data.copy_(
                    pool.buffer[b.offset : b.offset + b.size],
                    non_blocking=True,
                )
            offset += b.size
        assert offset == bucket.size, f"offset {offset} != bucket_size {bucket.size}"
        if owner_rank is not None:
            self._p2p_store.batch_transfer_sync_read(target_addr, buf_ptrs, remote_ptrs, lens)
        self.device_manager.device_module.synchronize()

    def init_process_group_for_ranks(
        self,
        ranks: list[int],
        *,
        master_port: int | None = None,
        timeout: timedelta = timedelta(minutes=10),
    ):
        """
        Initialize the process group for the ranks. This global group can be easily destroyed by calling dist.destroy_process_group.

        Args:
            ranks: The ranks to initialize the process group. ranks should be a subset of all ranks.
            master_port: The specified port of the master node. If not set, will use _get_master_port to get the port.
            timeout: The timeout of the process group.
        """
        assert not dist.is_initialized()
        assert ranks, "ranks should be set"
        if self._rank not in ranks:
            return
        assert self._all_hosts, "all_hosts should be set"
        assert len(self._all_hosts) == self._world_size // self._gpu_count, (
            f"world_size {self._world_size} should be equal to all_hosts {len(self._all_hosts)}"
        )
        rank = ranks.index(self._rank)
        master_addr = self._all_hosts[ranks[0] // self._gpu_count]
        master_port = _get_master_port(master_port)
        logger.info(
            f"[rank{self._rank}] start to init process group as virtual_rank {rank}, "
            f"master_addr {master_addr}, master_port {master_port}, world_size {len(ranks)}, "
        )
        # only initialize process group and store for ranks, other nodes are not initialized
        # and will not participate in this update. Since they have registered memory addresses
        # to p2p_store at the beginning, update ranks can directly get the memory addresses
        # from other nodes and put the weights into the buffer.
        store = dist.TCPStore(
            master_addr, master_port, len(ranks), is_master=rank == 0, timeout=timeout
        )
        dist.init_process_group(
            backend="nccl", world_size=len(ranks), rank=rank, timeout=timeout, store=store
        )

    def _get_addr_ptrs(self, owner_rank: int) -> tuple[str, list[tuple[int, int]]]:
        addr = self._current_global_parameter_metas[owner_rank].p2p_store_addr
        metas_list = self._current_global_parameter_metas[owner_rank].memory_buffer_metas_list
        return addr, [(metas.ptr, metas.size) for metas in metas_list]

    def _register_parameters_to_p2p_store(self, checkpoint_name: str):
        assert self._p2p_store is not None, "p2p store is not initialized"
        pool = self._memory_pool[checkpoint_name]
        if len(pool) == 0:
            return
        named_tensors, tensor_ptrs = {}, []
        for idx, memory_buffer in enumerate(pool):
            named_tensors[f"memory_pool_{checkpoint_name}_{idx}"] = memory_buffer.buffer
            tensor_ptrs.append((memory_buffer.buffer.data_ptr(), memory_buffer.size))
        self._p2p_store.register_named_tensors(named_tensors)

    def _unregister_parameters_from_p2p_store(self, checkpoint_name: str) -> int:
        assert self._p2p_store is not None, "p2p store is not initialized"
        pool = self._memory_pool[checkpoint_name]
        if len(pool) == 0:
            return 0
        return self._p2p_store.unregister_named_tensors(
            [f"memory_pool_{checkpoint_name}_{idx}" for idx, _ in enumerate(pool)]
        )

    def _update_per_bucket(
        self,
        checkpoint_name: str,
        req_func: Callable[[list[tuple[str, str]]], None],
        ranks: list[int] | None = None,
    ):
        assert len(self._current_global_parameter_metas) != 0, "parameter metas is empty"
        assert dist.is_initialized(), "process group is not initialized"
        # if both ranks is None or [], it will use fully broadcast to update to all ranks
        if not ranks:
            logger.info(f"[rank{self._rank}] update checkpoint {checkpoint_name}")
        # if ranks is set, it will use p2p to update to the ranks
        else:
            assert self._p2p_store is not None, "p2p store is not initialized"
            assert ranks, "ranks should be set"

            need_update = self._rank in ranks
            logger.info(
                f"[rank{self._rank}] update checkpoint {checkpoint_name} p2p, {need_update=} with {ranks=}, "
                f"gpu_count {self._gpu_count}, world_size {self._world_size}"
            )

            if not need_update:
                return
            # first execute a barrier to avoid subsequent cuda oom
            dist.barrier()

        bucket_size, disable_h2d_buffer = self._detect_bucket_size()
        buckets = _gen_h2d_buckets(
            self._current_global_parameter_metas,
            bucket_size,
            self._local_rdma_devices,
            self._remote_rdma_devices,
            ranks,
        )

        h2d_buffer: torch.Tensor | None = (
            None
            if disable_h2d_buffer
            else torch.empty(bucket_size, dtype=torch.uint8, device=self.device_manager.device_type)
        )
        # p2p store need to register h2d_buffer to let other ranks read
        if ranks:
            h2d_buffer_name = "__h2d_buffer__"
            if h2d_buffer is not None and self._p2p_store is not None:
                self._p2p_store.register_named_tensors({h2d_buffer_name: h2d_buffer})
        receiver_rank_buckets: list[tuple[int, H2DBucket]] = []
        for receiver_rank, owner_rank, bucket in buckets:
            if receiver_rank != self._rank:
                continue
            receiver_rank_buckets.append((owner_rank, bucket))

        buffer = torch.empty(
            bucket_size * 2, dtype=torch.uint8, device=self.device_manager.device_type
        )
        handle = reduce_tensor(buffer)

        buckets_by_receiver_rank: dict[int, list[H2DBucket]] = defaultdict(list)
        max_len = 0
        for receiver_rank, _, bucket in buckets:
            buckets_by_receiver_rank[receiver_rank].append(bucket)
            if len(buckets_by_receiver_rank[receiver_rank]) > max_len:
                max_len = len(buckets_by_receiver_rank[receiver_rank])

        socket, socket_paths = self._bind_zmq_socket()
        req_thread = threading.Thread(
            target=req_func,
            args=(socket_paths,),
        )
        req_thread.start()
        socket.send_pyobj(handle)

        gidx = 0
        bcast_rank_map = _get_bcast_rank_map(self._world_size, ranks)
        for i in range(max_len):
            if i < len(receiver_rank_buckets) and not disable_h2d_buffer:
                self._copy_to_buffer(
                    checkpoint_name,
                    receiver_rank_buckets[i][1],
                    h2d_buffer,
                    receiver_rank_buckets[i][0] if ranks else None,
                )
            for receiver_rank, _buckets in buckets_by_receiver_rank.items():
                if i >= len(_buckets):
                    continue
                bucket = _buckets[i]
                alloc, reserved = (
                    self.device_manager.device_module.memory_allocated() / 1024 / 1024,
                    self.device_manager.device_module.memory_reserved() / 1024 / 1024,
                )
                self._logger_rank0(
                    f"[rank{self._rank}] begin to update bucket {gidx + 1}/{len(buckets)} receiver_rank {receiver_rank} in checkpoint {checkpoint_name}, bucket_size: {bucket.size / 1024 / 1024:.2f}MiB, length: {len(bucket.items)}. "
                    f"Current CUDA allocated {alloc:.2f} MB, "
                    f"reserved {reserved:.2f} MB."
                )
                start = gidx % 2 * bucket_size
                buffer_b: torch.Tensor = buffer[start : start + bucket.size]
                if receiver_rank == self._rank:
                    if disable_h2d_buffer:
                        self._copy_to_buffer(checkpoint_name, bucket, buffer_b)
                    else:
                        buffer_b.data.copy_(h2d_buffer[: bucket.size])
                brank = bcast_rank_map[receiver_rank]
                dist.broadcast(buffer_b, src=brank)
                socket.recv()
                dist.barrier()
                socket.send_pyobj(_to_named_tensor(bucket.items, gidx % 2 * bucket_size))
                gidx += 1

        socket.recv()
        socket.send_pyobj(None)
        socket.recv()
        req_thread.join()
        dist.barrier()
        socket.close()
        if ranks and h2d_buffer is not None:
            self._p2p_store.unregister_named_tensors([h2d_buffer_name])

        self.device_manager.device_module.empty_cache()
