//===- AIECoreMemory.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared vocabulary for a core tile's data memory layout, used by the places
// that have to agree about it: the buffer allocator (AIEAssignBuffers) and
// the linker script emitter (AIETargetLdScript).
//
// Names shared between the dialect and the aiecc driver -- the outlined core
// frame symbol, the computed-stack-requirement attribute -- live in
// AIECoreSymbols.h instead; they are naming contracts, not memory layout.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_IR_AIECOREMEMORY_H
#define AIE_DIALECT_AIE_IR_AIECOREMEMORY_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace xilinx::AIE {

// A half-open [start, start + size) run of bytes.
struct MemoryRun {
  int64_t start = 0;
  int64_t size = 0;
};

// Largest run of free bytes in [0, memSize), given occupied half-open
// intervals that need not be sorted or disjoint.
//
// This is the number that decides whether a core links. AIETargetLdScript
// hands the core compiler exactly one region for its own .data/.rodata/.bss --
// this run -- so the allocator's reserved_data_size acceptance test and the
// linker script must agree byte for byte. Both call this rather than sweeping
// the intervals themselves.
inline MemoryRun
largestFreeRun(int64_t memSize,
               llvm::SmallVector<std::pair<int64_t, int64_t>> occupied) {
  llvm::sort(occupied);
  MemoryRun best;
  int64_t cursor = 0;
  for (auto &interval : occupied) {
    // A zero-length interval (a zero-sized buffer) occupies no bytes, so it
    // must not act as a split point in the middle of an otherwise-contiguous
    // run.
    if (interval.first == interval.second)
      continue;
    int64_t gapEnd = std::min(interval.first, memSize);
    if (gapEnd - cursor > best.size)
      best = {cursor, gapEnd - cursor};
    cursor = std::max(cursor, interval.second);
    if (cursor >= memSize)
      return best;
  }
  if (memSize - cursor > best.size)
    best = {cursor, memSize - cursor};
  return best;
}

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_IR_AIECOREMEMORY_H
