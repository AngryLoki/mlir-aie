//===- bank_aware_reserved_data.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core's own compiled sections (.data/.rodata/.bss) share data memory with
// the buffers placed here, and the generated linker script hands the core
// compiler exactly one region. Spreading buffers over banks chops that space
// up, so `reserved_data_size` tells the allocator how much contiguous room the
// core still needs.
//
// The allocator carves that room out *before* placing the unconstrained
// buffers, choosing the spot that leaves the largest single run behind. So a
// reservation no longer costs bank spreading: it takes a contiguous block and
// the buffers spread over whatever is left, rather than everything packing
// into one bank to leave a big gap at the far end.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// Without a reservation the three buffers spread round-robin over banks 0-2,
// which is what limits DMA bank contention. The largest surviving gap is one
// bank, 16384 bytes.
// CHECK-LABEL: module @no_reservation_spreads
// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c"} : memref<4096xi8>
// CHECK: data_length = 28672 : i32, data_origin = 36864 : i32
module @no_reservation_spreads {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// A reservation big enough that it cannot coexist with a spread placement is
// carved out of the bottom of memory, and the three buffers then spread over
// the banks above it. The old allocator answered this by packing all three
// into bank 0 and leaving the top 52224 bytes free; carving the block out
// first gets the core its 40000 bytes *and* keeps two banks of spread.
// CHECK-LABEL: module @reservation_carves_out_a_block
// CHECK: %a = aie.buffer(%tile_0_2) {address = 41024 : i32, mem_bank = 2 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 53248 : i32, mem_bank = 3 : i32, sym_name = "c"} : memref<4096xi8>
// The grant here is exactly the request: the buffers took everything above it,
// so there was no unused space left to fold back in.
// CHECK: data_length = 40000 : i32, data_origin = 1024 : i32
module @reservation_carves_out_a_block {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 40000 : i32}
  }
}

// -----

// A small reservation leaves the round-robin spread over banks 0-2 intact;
// only `a` slides up within bank 0, to sit flush above the block reserved at
// the bottom rather than stranding it.
// CHECK-LABEL: module @small_reservation_keeps_spread
// CHECK: %a = aie.buffer(%tile_0_2) {address = 9216 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<4096xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<4096xi8>
// CHECK: %c = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "c"} : memref<4096xi8>
// CHECK: data_length = 28672 : i32, data_origin = 36864 : i32
module @small_reservation_keeps_spread {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 8192 : i32}
  }
}
