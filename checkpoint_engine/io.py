import os
import pickle
from typing import Any, BinaryIO

import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel
from safetensors.torch import safe_open

from checkpoint_engine.types import ParameterMeta


def _load_checkpoint_file(file_path: str) -> tuple[int, dict[str, tuple["FileMeta", torch.Tensor]]]:
    def _safetensors_load(fn: str) -> dict[str, tuple["FileMeta", torch.Tensor]]:
        ret = {}
        with safe_open(fn, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                weight = f.get_tensor(name)
                meta = {
                    "key": name,
                    "dtype": weight.dtype,
                    "shape": weight.shape,
                    "type": type(weight),
                    "tp_concat_dim": -1,  # safetensors does not support tp_concat_dim
                }
                ret[name] = (meta, weight)
        return ret

    # deprecated, will be removed in the future
    def _fast_np_load(fn: str) -> dict[str, tuple["FileMeta", torch.Tensor]]:
        """load *.np file and return memmap and related tensor meta"""

        def parse_npy_header(fin: BinaryIO) -> dict[str, Any]:
            start = fin.tell()
            major, minor = np.lib.format.read_magic(fin)
            if major == 1 and minor == 0:
                read_header_fn = np.lib.format.read_array_header_1_0
            elif major == 2 and minor == 0:
                read_header_fn = np.lib.format.read_array_header_2_0
            else:
                raise ValueError(
                    f"unknown version {major}.{minor} when parsing npy header from {fn}"
                )
            shape, is_fortran, dtype = read_header_fn(fin)
            return {
                "shape": shape,
                "is_fortran": is_fortran,
                "dtype": dtype,
                "header_length": fin.tell() - start,
            }

        meta_fn = fn + ".meta"
        with open(meta_fn, "rb") as fin:
            meta_lst = pickle.load(fin)

        tensors = []
        offset = 0
        with open(fn, "rb") as fin:
            fin.seek(0, os.SEEK_END)
            filesize = fin.tell()
            fin.seek(0)
            while fin.tell() < filesize:
                tensor_meta = parse_npy_header(fin)
                tensor = np.memmap(
                    fn,
                    dtype=tensor_meta["dtype"],
                    mode="c",
                    offset=offset + tensor_meta["header_length"],
                    shape=tensor_meta["shape"],
                )
                offset += tensor_meta["header_length"] + tensor.nbytes
                fin.seek(offset)
                tensors.append(tensor)

        assert len(meta_lst) == len(tensors)
        ret = {}
        for meta, tensor in zip(meta_lst, tensors):
            if meta["type"] == torch.Tensor:
                tensor = torch.from_numpy(tensor)
            tensor = tensor.view(dtype=meta["dtype"]).view(*meta["shape"])
            ret[meta["key"]] = (meta, tensor)
        return ret

    tp_rank = 0
    if file_path.endswith(".npy"):
        logger.warning("numpy model file is deprecated, will be removed in the future")
        filename_split = os.path.basename(file_path).split(".")
        # if using numpy and want to specify tp rank
        # file should be in model.{layer}.{tp}[.{ep}].npy format
        tp_rank = int(filename_split[2]) if len(filename_split) > 3 else 0
        ret = _fast_np_load(file_path)
    elif file_path.endswith(".safetensors"):
        ret = _safetensors_load(file_path)
    else:
        raise ValueError(f"unsupported file format: {file_path}")
    return tp_rank, ret


def _concat_tp_weights(
    tp_weights: list[torch.Tensor], tp_concat_dim: int, tp_size: int
) -> torch.Tensor:
    """Concat tp weights with meta info.
    If meta.concat_dim is -1, meas this is shared tp weights, just use the first weights.
    Else we will cat weights in concat_dim.
    """
    if tp_concat_dim == -1:
        return tp_weights[0]
    assert tp_size == len(tp_weights)
    if len(tp_weights) == 1:
        return tp_weights[0]
    return torch.cat([w for w in tp_weights], dim=tp_concat_dim)

def _load_checkpoint(files: list[str]) -> dict[str, torch.Tensor]:
    class TPMeta(BaseModel):
        concat_dim: int
        size: int

    parameters: dict[str, torch.Tensor] = {}
    parameter_metas: dict[str, ParameterMeta] = {}
    tp_metas: dict[str, TPMeta] = {}
    parameters_with_tp: dict[str, dict[int, torch.Tensor]] = {}
    for file in files:
        tp_rank, ret = _load_checkpoint_file(file)
        for parameter_name, (meta, weight) in ret.items():
            if parameter_name not in parameters_with_tp:
                parameters_with_tp[parameter_name] = {}
            parameters_with_tp[parameter_name][tp_rank] = weight
            if parameter_name not in tp_metas:
                tp_metas[parameter_name] = TPMeta(
                    concat_dim=meta["tp_concat_dim"],
                    size=1,
                )
            if parameter_name not in parameter_metas:
                assert isinstance(meta["dtype"], torch.dtype), (
                    f"meta {meta} dtype should be torch.dtype"
                )
                assert isinstance(meta["shape"], torch.Size), (
                    f"meta {meta} shape should be torch.Size"
                )
                parameter_metas[parameter_name] = ParameterMeta(
                    name=parameter_name,
                    shape=meta["shape"],
                    dtype=meta["dtype"],
                )
            tp_meta = tp_metas[parameter_name]
            if tp_meta.concat_dim != -1:
                tp_meta.size = max(tp_meta.size, tp_rank + 1)
    for name, tp_meta in tp_metas.items():
        if tp_meta.concat_dim != -1:
            shape = list(parameter_metas[name].shape)
            shape[tp_meta.concat_dim] = shape[tp_meta.concat_dim] * tp_meta.size
            parameter_metas[name] = ParameterMeta(
                name=name, shape=torch.Size(shape), dtype=parameter_metas[name].dtype
            )
        weights_in_cpu = [parameters_with_tp[name][key] for key in sorted(parameters_with_tp[name])]
        # TODO: here concat is serial, which may be slow
        # but since tp storage is not used in the future
        # we ignore this performance issue for now
        parameters[name] = _concat_tp_weights(weights_in_cpu, tp_meta.concat_dim, tp_meta.size)
    for name, parameter in parameters.items():
        assert name in parameter_metas, f"parameter {name} not found in parameter_metas"
        assert parameter_metas[name].shape == parameter.shape, (
            f"parameter {name} shape mismatch, {parameter_metas[name].shape} != {parameter.shape}"
        )
        assert parameter_metas[name].dtype == parameter.dtype, (
            f"parameter {name} dtype mismatch, {parameter_metas[name].dtype} != {parameter.dtype}"
        )
    return parameters
