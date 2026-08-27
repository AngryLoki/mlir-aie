//===- reserved_data_size_bcf.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Chess links against a BCF, not the ldscript AIETargetLdScript emits: every
// placed buffer is marked `_reserved DMb <addr> <size>`, so chess's linker
// sees exactly the same holes the allocator computed, whichever ones survive.
// This is claimed to be "conservative but never insufficient" for chess
// without any BCF-side change, so pin down that the BCF actually reflects a
// bank-aware placement chosen to satisfy reserved_data_size: the reservation
// is carved out of the bottom of the tile and the three 4 KB buffers are
// placed above it. aie-translate --aie-generate-bcf needs no chess toolchain
// or hardware -- it is a pure MLIR-to-text translation.
//
// This is also where the chess flow gets the reservation for free. BCF has no
// data-region concept: it marks each buffer `_reserved` and lets the chess
// linker use whatever is left. Because the reservation now manifests as a real
// gap in the buffer addresses rather than a number the allocator merely
// checked, that leftover space is genuinely there.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 | FileCheck %s

// CHECK: _symbol a 0x7A040 4096
// CHECK: _reserved DMb 0x7A040 4096
// CHECK: _symbol b 0x7C000 4096
// CHECK: _reserved DMb 0x7C000 4096
// CHECK: _symbol c 0x7D000 4096
// CHECK: _reserved DMb 0x7D000 4096

module {
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
