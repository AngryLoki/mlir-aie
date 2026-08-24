//===- reserved_data_size_mixed_measurable_warns.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Only a plain relocatable object can be measured; an archive or a path that
// does not resolve to anything is skipped rather than approximated (see
// measureObjectDataSectionBytes / populateReservedDataSize). One measurable
// object alongside an archive and a missing file: reserved_data_size is
// measured from what could be read, and a warning names exactly the
// artifact(s) that could not be.

// REQUIRES: peano
// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: clang++ --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG -c %S/reserved_data_size_measured_kernel.cc -o %t.d/reserved_data_size_measured_kernel.o
// RUN: echo not-a-real-object > %t.d/dummy_member.txt
// RUN: llvm-ar rcs %t.d/unmeasurable.a %t.d/dummy_member.txt
// RUN: cd %t.d && aiecc --cut='reserved_data.mlir' --checkpoint=%t.d/ckpt %s 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: cat %t.d/ckpt/*/reserved_data.mlir | FileCheck %s

// DIAG: reserved_data_size auto-measured as 8448 bytes from link_files, but could not inspect 2 artifact(s) (archive, bitcode, or unreadable), so this estimate may be incomplete: missing.o, unmeasurable.a

// CHECK: reserved_data_size = 8448 : i32

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.core(%tile_0_2) {
      aie.end
    } { link_files = ["reserved_data_size_measured_kernel.o", "missing.o", "unmeasurable.a"] }
  }
}
