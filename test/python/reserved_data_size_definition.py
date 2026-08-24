# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# Pins down the IRON-side contract for Worker(reserved_data_size=): the
# requested value survives resolution and appears as the reserved_data_size
# attribute on the emitted aie.core, including the "0 is a legal explicit
# value" case, both through the Worker convenience class and through the
# lower-level @core(...) dialect decorator.

from aie.iron import Program, Runtime, Worker

from aie.iron.device import NPU1

# CHECK: {reserved_data_size = 4096 : i32
my_worker = Worker(None, reserved_data_size=4096, while_true=False)


def sequence():
    pass


rt = Runtime(sequence, [])

my_program = Program(NPU1(), rt, workers=[my_worker])

module = my_program.resolve_program()

print(module)


# CHECK: {reserved_data_size = 0 : i32

from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx


def mlir_aie_design():

    @device(AIEDevice.npu1)
    def device_body():

        ComputeTile1 = tile(0, 2)

        @core(ComputeTile1, reserved_data_size=0)
        def core_body():
            pass


with mlir_mod_ctx() as ctx:
    mlir_aie_design()
    print(ctx.module)


# Worker validates reserved_data_size the same way it validates other
# int-typed knobs: a clear ValueError at construction time rather than an
# opaque MLIR verifier failure much later in the pipeline.
try:
    Worker(None, reserved_data_size=-1, while_true=False)
    raise AssertionError("expected ValueError for negative reserved_data_size")
except ValueError:
    pass

try:
    Worker(None, reserved_data_size="4096", while_true=False)
    raise AssertionError("expected ValueError for non-int reserved_data_size")
except ValueError:
    pass

try:
    Worker(None, reserved_data_size=True, while_true=False)
    raise AssertionError("expected ValueError for bool reserved_data_size")
except ValueError:
    pass

# CHECK: PASS
print("PASS")
