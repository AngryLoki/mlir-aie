//===- bank_aware_reserved_data_no_free_run.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The boundary case for largestFreeRun(): a tile with every byte already
// spoken for by the stack and one buffer, so the largest free run is exactly
// 0. reserved_data_size = 0 must still succeed (0 bytes always fit in a
// 0-byte run); anything above 0 must fail. See
// bank_aware_reserved_data_no_free_run_error.mlir for the failure case.
//
// tile(0, 2) on npu2 has 65536 bytes; the 1024-byte stack plus "a"'s 64512
// unaligned bytes leave nothing over.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, aligned = false, mem_bank = 0 : i32, sym_name = "a"} : memref<64512xi8>
module @zero_free_run_zero_reservation {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a", aligned = false} : memref<64512xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 0 : i32}
  }
}
