//===- reserved_data_size_all_unmeasurable_absent.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// When every link_files entry is unmeasurable (an archive, or a path that
// does not resolve to anything), reserved_data_size is left absent entirely
// -- "unknown, reserve nothing", the same as a core with no link_files.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: echo not-a-real-object > %t.d/dummy_member.txt
// RUN: llvm-ar rcs %t.d/unmeasurable.a %t.d/dummy_member.txt
// RUN: cd %t.d && aiecc --cut='reserved_data.mlir' --checkpoint=%t.d/ckpt %s
// RUN: cat %t.d/ckpt/*/reserved_data.mlir | FileCheck %s

// CHECK: aie.core
// CHECK-NOT: reserved_data_size =

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.core(%tile_0_2) {
      aie.end
    } { link_files = ["missing.o", "unmeasurable.a"] }
  }
}
