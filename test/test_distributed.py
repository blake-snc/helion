from __future__ import annotations

from datetime import timedelta

import torch
from torch import Tensor
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from helion._testing import TestCase
from helion._testing import all_gather_object
from helion._testing import onlyBackends
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
from helion._utils import sync_seed


import torch
import torch.distributed as dist
import contextlib
import unittest
import os
from torch.testing._internal.common_distributed import MultiProcessTestCase
from datetime import timedelta
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import run_tests
from helion._testing import onlyBackends
from torch.utils.cpp_extension import load_inline
import helion.language as hl
import helion
from helion._utils import sync_seed
from helion._testing import TestCase
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
from helion._testing import code_and_output
from helion._testing import all_gather_object
import torch.distributed._symmetric_memory as symm_mem

def one_shot_all_reduce_kernel(
    a_shared: torch.Tensor,
    my_rank: hl.constexpr,
    group_name: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
) -> torch.Tensor:
    out = torch.empty_like(a_shared)
    N = out.size(0)
    a_shared_tuple = torch.ops.symm_mem.get_remote_tensors(a_shared, group_name)

    for tile_n in hl.tile(N):
        acc = hl.zeros([tile_n], dtype=a_shared.dtype, device=a_shared.device)

        for a in a_shared_tuple:
            acc += a[tile_n]

        out[tile_n] = acc
    return out


@onlyBackends(["triton"])
@instantiate_parametrized_tests
class TestDistributed(TestCase, MultiProcessTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._class_stack = contextlib.ExitStack()
        cls._class_stack.enter_context(unittest.mock.patch.dict(os.environ, {
            "CHECK_CONFIG_CONSISTANCY": "1",
        }))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_stack.close()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        super().tearDown()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=60), dist.group.WORLD
        )
        torch.manual_seed(42 + self.rank)

    def _cleanup_process(self):
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    def test_sync_seed(self):
        def _all_eq(xlist: list[Tensor]) -> bool:
            assert len(xlist) > 1
            lhs = xlist[0]
            return all(torch.allclose(lhs.cpu(), rhs.cpu()) for rhs in xlist[1:])

        self._init_process()
        torch.manual_seed(42 + self.rank)

        x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x)

        self.assertFalse(_all_eq(xlist))

        with sync_seed():
            x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x)
        self.assertTrue(_all_eq(xlist))

        x = torch.randn(1024, device=self.device)
        xlist = all_gather_object(x)
        self.assertFalse(_all_eq(xlist))

        self._cleanup_process()

   

    @skipIfRocm("Distributed example requires CUDA/NCCL")
    @skipIfXPU("Distributed operations require CCL, not yet fully integrated")
    @skip_if_lt_x_gpu(4)
    @parametrize("autotuner", ["fixed", "PatternSearch", "LFBOPatternSearch", "LFBOTreeSearch", "DifferentialEvolutionSearch", "DESurrogateHybrid", "FiniteSearch", "RandomSearch"])
    def test_all_reduce(self, autotuner):
        self._init_process()
        if autotuner == "fixed":
            kernel = helion.kernel(
                config=helion.Config(
                    block_sizes=[8192],
                    num_warps=32,
                ),
            )(one_shot_all_reduce_kernel)
            context = contextlib.nullcontext()
        elif autotuner == "FiniteSearch":
            kernel = helion.kernel(
                configs=[
                    helion.Config(
                        block_sizes=[8192],
                        num_warps=32
                    ),
                    helion.Config(
                        block_sizes=[4096],
                        num_warps=32
                    ),
                ]
            )(one_shot_all_reduce_kernel)
            context = unittest.mock.patch.dict(os.environ, {"HELION_AUTOTUNER": autotuner})
        else:
            kernel = helion.kernel(one_shot_all_reduce_kernel)
            context = unittest.mock.patch.dict(os.environ, {"HELION_AUTOTUNER": autotuner})

        with context:
            self.do_test_all_reduce(kernel)

        self._cleanup_process()

    def do_test_all_reduce(self, kernel):
        group = dist.group.WORLD

        N = 16384
        dtype = torch.bfloat16

        a_shared = symm_mem.empty(
            N, dtype=dtype, device=self.device
        ).normal_()

        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=group)

        result = kernel(
            a_shared,
            symm_mem_hdl.rank,
            group.group_name,
            symm_mem_hdl.world_size,
        )

        torch.cuda.synchronize()

        expected = torch.empty_like(result).copy_(a_shared)
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

if __name__ == "__main__":
    run_tests()
