//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIECoreMemory.h"
#include "aie/Dialect/AIE/IR/AIECoreSymbols.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"

#include "llvm/ADT/BitVector.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEASSIGNBUFFERADDRESSES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-assign-buffers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// Surfaced in the memory-map diagnostics below so a user sees the call-graph
// analysis's number alongside the stack region, reserved_data_size, and free
// space in one place, instead of cross-referencing a warning emitted earlier
// in the build log. Absent unless aiecc ran its stack analysis.
static std::optional<int64_t> getComputedStackRequirement(TileOp tile) {
  CoreOp core = tile.getCoreOp();
  if (!core)
    return std::nullopt;
  if (auto attr =
          core->getAttrOfType<IntegerAttr>(kComputedStackRequirementAttrName))
    return attr.getInt();
  return std::nullopt;
}

// The two numbers every allocation scheme needs before it can place anything:
// how much data memory the tile has, and what alignment its load/store bus
// requires. A memtile and a compute tile ask the target model for these
// differently.
struct TileMemoryLimits {
  int64_t maxDataMemorySize;
  uint32_t tileAlignBitWidth;
};
static TileMemoryLimits tileMemoryLimits(TileOp tile,
                                         const AIETargetModel &targetModel) {
  if (tile.isMemTile())
    return {targetModel.getMemTileSize(),
            targetModel.getMemTileLoadStoreBusWidth()};
  return {targetModel.getLocalMemorySize(),
          targetModel.getComputeTileLoadStoreBusWidth()};
}

// Every buffer here must already have an address: called once allocation has
// either succeeded or is about to be reported failed, to put the memory map
// and the final overlap/overflow checks in address order.
static void sortBuffersByAddress(SmallVectorImpl<BufferOp> &buffers) {
  llvm::sort(buffers, [](BufferOp a, BufferOp b) {
    assert(a.getAddress().has_value() && "buffer must have address assigned");
    assert(b.getAddress().has_value() && "buffer must have address assigned");
    return a.getAddress().value() < b.getAddress().value();
  });
}

// The region the core compiler gets for its own sections, computed the way
// AIETargetLdScript's fallback computes the `data` region. Both call the
// shared largestFreeRun on the same interval list, so the value stamped on the
// CoreOp and the value that emitter would derive cannot drift apart.
static MemoryRun coreDataRun(int64_t memSize, int64_t stackSize,
                             ArrayRef<BufferOp> buffers) {
  SmallVector<std::pair<int64_t, int64_t>> occupied;
  occupied.emplace_back(0, stackSize);
  for (auto buffer : buffers)
    if (auto addr = buffer.getAddress())
      occupied.emplace_back(*addr, *addr + buffer.getAllocationSize());
  return largestFreeRun(memSize, std::move(occupied));
}

// Record where the core's data region ended up, for AIETargetLdScript to emit
// verbatim. No-op on a tile with no core: memtiles and shims have no compiled
// sections of their own, and no linker script is generated for them.
static void stampCoreDataRegion(TileOp tile, MemoryRun run) {
  CoreOp core = tile.getCoreOp();
  if (!core)
    return;
  Builder b(tile.getContext());
  core.setDataOriginAttr(b.getI32IntegerAttr(run.start));
  core.setDataLengthAttr(b.getI32IntegerAttr(run.size));
}

// Called at the top of each scheme so a stamp from an earlier run -- or from
// an attempt that then failed -- is never left behind describing a placement
// that no longer exists.
static void clearCoreDataStamp(TileOp tile) {
  if (CoreOp core = tile.getCoreOp()) {
    core->removeAttr("data_origin");
    core->removeAttr("data_length");
  }
}

static bool isBufferPreAllocated(BufferOp buffer) {
  auto addr = buffer.getAddress();
  auto memBank = buffer.getMemBank();
  return (addr != std::nullopt || memBank != std::nullopt);
}

// Return an address that is aligned to tile's load/store bus
// NOTE: assume address are byte address
static int64_t getAlignedAddress(int64_t address, uint32_t alignBitWidth) {
  assert(alignBitWidth != 0 && alignBitWidth % 8 == 0 &&
         "alignBitWidth must be a non-zero multiple of 8");
  uint32_t alignByteWidth = alignBitWidth / 8;
  if (address % alignByteWidth == 0) {
    return address;
  }
  return ((address / alignByteWidth) + 1) * alignByteWidth;
}

// Check that every buffer in the list is properly aligned (when its
// `aligned` attribute is set) and that no two buffers overlap. The input
// vector must be sorted by ascending address. Emits an error on the first
// offending buffer and returns false; returns true otherwise.
static bool checkAndPrintBufferOverlap(ArrayRef<BufferOp> sortedBuffers,
                                       uint32_t tileAlignBitWidth) {
  uint32_t alignByteWidth = tileAlignBitWidth / 8;
  BufferOp prev = nullptr;
  for (auto cur : sortedBuffers) {
    auto curAddrOpt = cur.getAddress();
    assert(curAddrOpt.has_value() && "buffer must have address assigned");
    int64_t curAddr = *curAddrOpt;

    // Alignment check.
    if (cur.getAligned() && alignByteWidth != 0 &&
        curAddr % alignByteWidth != 0) {
      cur.emitOpError("buffer '")
          << cur.name() << "' at address 0x" << llvm::utohexstr(curAddr)
          << " is not aligned to tile load/store bus width ("
          << tileAlignBitWidth << " bits)";
      return false;
    }

    // A zero-sized buffer covers no bytes, so it cannot overlap anything; it
    // would otherwise report a false overlap when it shares an address.
    if (cur.getAllocationSize() == 0)
      continue;

    // Overlap check against the closest buffer below in address order.
    if (prev) {
      auto prevAddrOpt = prev.getAddress();
      assert(prevAddrOpt.has_value() && "buffer must have address assigned");
      int64_t prevAddr = *prevAddrOpt;
      int64_t prevEnd = prevAddr + prev.getAllocationSize();
      if (curAddr < prevEnd) {
        cur.emitOpError("buffer '")
            << cur.name() << "' at address 0x" << llvm::utohexstr(curAddr)
            << " overlaps with '" << prev.name() << "' at address 0x"
            << llvm::utohexstr(prevAddr)
            << " (size: " << prev.getAllocationSize() << " bytes)";
        return false;
      }
    }
    prev = cur;
  }
  return true;
}

// Check if there is any overlap between the stack and the allocated buffers.
static bool checkAndPrintOverlapStackframe(int64_t stacksize,
                                           ArrayRef<BufferOp> buffers) {
  for (auto buf : buffers) {
    // A zero-sized buffer covers no bytes, so it cannot overlap the stack.
    if (buf.getAllocationSize() == 0)
      continue;
    auto bufAddrOpt = buf.getAddress();
    assert(bufAddrOpt.has_value() && "buffer must have address assigned");
    int64_t bufAddr = *bufAddrOpt;
    if (bufAddr < stacksize) {
      buf.emitOpError("buffer '")
          << buf.name() << "' at address 0x" << llvm::utohexstr(bufAddr)
          << " overlaps with stack (size: " << stacksize << " bytes)";
      return false;
    }
  }
  return true;
}

// A reservation the packing can't honour must fail here too, not just in
// bank-aware: a design whose bank-aware attempt fails for an unrelated reason
// falls back to basic-sequential, and the shortfall would otherwise surface
// much later as a linker region-overflow error naming neither the tile nor
// the buffers responsible.
static bool checkAndPrintReservedData(TileOp tile, int64_t freeRun,
                                      int64_t reservedData) {
  if (freeRun >= reservedData)
    return true;
  tile.emitWarning("buffers leave only ")
      << freeRun
      << " contiguous bytes for the core's data sections, which need "
      << reservedData << " bytes.";
  return false;
}

// One line of a memory-map diagnostic: "<indent>name \t: 0xADDR-0xEND \t(N
// bytes)<suffix>". Shared by the two diagnostics below so their formatting
// can't drift apart.
static void printMemoryMapEntry(Diagnostic &note, StringRef name,
                                int64_t address, int64_t size, int indent,
                                StringRef suffix = "") {
  for (int i = 0; i < indent; ++i)
    note << "\t";
  note << name << " \t"
       << ": 0x" << llvm::utohexstr(address) << "-0x"
       << llvm::utohexstr(address + size - 1) << " \t(" << size << " bytes)"
       << suffix << "\n";
}

//===----------------------------------------------------------------------===//
// BasicAllocation : sequential alloc from largest to smallest
//===----------------------------------------------------------------------===//
static bool checkAndPrintOverflow(TileOp tile, int64_t address,
                                  int64_t maxDataMemorySize, int64_t stacksize,
                                  ArrayRef<BufferOp> buffers) {
  if (address > maxDataMemorySize) {
    InFlightDiagnostic error =
        tile.emitOpError("allocated buffers exceeded available memory\n");
    auto &note = error.attachNote() << "MemoryMap:\n";
    auto printbuffer = [&](StringRef name, int64_t address, int64_t size) {
      printMemoryMapEntry(note, name, address, size, /*indent=*/1);
    };
    if (stacksize > 0)
      printbuffer("(stack)", 0, stacksize);
    else
      note << "\t(no stack allocated)\n";
    if (auto computed = getComputedStackRequirement(tile))
      note << "\t(call-graph analysis computed this core's callees need >= "
           << *computed << " bytes of stack)\n";

    for (auto buffer : buffers) {
      auto bufferAddrOpt = buffer.getAddress();
      assert(bufferAddrOpt.has_value() && "buffer must have address assigned");
      printbuffer(buffer.name(), *bufferAddrOpt, buffer.getAllocationSize());
    }
    return false;
  }
  return true;
}

static bool basicAllocation(TileOp tile) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device)
    return false;

  clearCoreDataStamp(tile);

  auto [maxDataMemorySize, tileAlignBitWidth] =
      tileMemoryLimits(tile, getTargetModel(tile));

  SmallVector<BufferOp> buffers;
  SmallVector<BufferOp> allocated_buffers;
  SmallVector<BufferOp> allBuffers_on_tile;
  // Collect all the buffers for this tile. If the buffer has an address, add
  // it to allocated_buffers. Otherwise, add it to buffers.
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile) {
      if (buffer.getAddress()) {
        allocated_buffers.push_back(buffer);
      } else {
        // Basic-sequential ignores mem_bank entirely (the bit-packed bump
        // pointer below has no notion of banks), so a mem_bank left over
        // here -- whether requested directly under this scheme, or rolled
        // back by a bank-aware attempt that failed and fell back to this one
        // -- describes a placement this scheme is not going to produce.
        // Clear it rather than let the address this scheme does assign carry
        // a bank label that may no longer be true.
        buffer->removeAttr("mem_bank");
        buffers.push_back(buffer);
      }
      allBuffers_on_tile.push_back(buffer);
    }
  });

  // Sort buffers by allocation size.
  std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
    return a.getAllocationSize() > b.getAllocationSize();
  });

  // Sort allocated_buffers by address
  std::sort(allocated_buffers.begin(), allocated_buffers.end(),
            [](BufferOp a, BufferOp b) {
              return a.getAddress().value() < b.getAddress().value();
            });

  // Address range owned by the MemTile is 0x80000.
  // Address range owned by the tile is 0x8000 in
  // AIE1 and 0x10000 in AIE2, but we need room at
  // the bottom for stack.
  int64_t stacksize = 0;
  int64_t address = 0;
  int64_t reservedData = 0;
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getEffectiveStackSize();
    address += stacksize;
    // Space the core's own compiled sections need; see reserved_data_size.
    reservedData = core.getReservedDataSize().value_or(0);
  }

  // Ensure alignment of preallocated buffer
  for (auto buffer : allocated_buffers) {
    auto bufferAddrOpt = buffer.getAddress();
    assert(bufferAddrOpt.has_value() &&
           "allocated_buffers only holds buffers with an address");
    if (buffer.getAligned() && *bufferAddrOpt % (tileAlignBitWidth / 8) != 0) {
      buffer.emitOpError("pre-allocated address must be aligned to tile "
                         "load/store bus width when aligned attribute is set");
      return false;
    }
  }

  // As the next address to allocate is assigned, skip over any buffers
  // from the allocated_buffers list.
  // Note: alignment must be applied *before* (and after each skip in) the
  // overlap-skip loop below, so the loop reasons about the buffer's actual
  // placement. Otherwise an unaligned candidate address can appear to fit
  // before the next pre-allocated buffer, but get bumped forward by
  // getAlignedAddress and silently alias that pre-allocated buffer.
  auto *current_alloc = allocated_buffers.begin();
  for (auto buffer : buffers) {
    assert(!buffer.getAddress());
    if (buffer.getAligned())
      address = getAlignedAddress(address, tileAlignBitWidth);
    while (current_alloc != allocated_buffers.end() &&
           address + buffer.getAllocationSize() >
               current_alloc->getAddress().value()) {
      address = current_alloc->getAddress().value() +
                current_alloc->getAllocationSize();
      if (buffer.getAligned())
        address = getAlignedAddress(address, tileAlignBitWidth);
      current_alloc++;
    }

    buffer.setAddress(address);
    address += buffer.getAllocationSize();
  }

  // Sort by smallest address before printing memory map and running the
  // overlap / overflow checks below.
  sortBuffersByAddress(allBuffers_on_tile);

  // Compute the true high-water mark across *all* buffers (including
  // pre-allocated ones above the dynamic-allocation cursor) so that
  // checkAndPrintOverflow sees a correct memory bound.
  int64_t highWater = address;
  if (!allBuffers_on_tile.empty()) {
    auto &last = allBuffers_on_tile.back();
    auto lastAddrOpt = last.getAddress();
    assert(lastAddrOpt.has_value() && "buffer must have address assigned");
    highWater =
        std::max<int64_t>(highWater, *lastAddrOpt + last.getAllocationSize());
  }

  // Check if memory was exceeded or buffers overlap, and print debug info.
  MemoryRun dataRun =
      coreDataRun(maxDataMemorySize, stacksize, allBuffers_on_tile);
  if (!checkAndPrintOverlapStackframe(stacksize, allBuffers_on_tile) ||
      !checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth) ||
      !checkAndPrintOverflow(tile, highWater, maxDataMemorySize, stacksize,
                             allBuffers_on_tile) ||
      !checkAndPrintReservedData(tile, dataRun.size, reservedData))
    return false;
  stampCoreDataRegion(tile, dataRun);
  return true;
}

//===----------------------------------------------------------------------===//
// SimpleBankAwareAllocation : round-robin each alloc over available banks
//===----------------------------------------------------------------------===//
namespace {
using BankLimits = struct BankLimits {
  int64_t startAddr;
  int64_t endAddr;
};
} // namespace

// Function that given a number of banks and their size, computes
// the start and end addresses for each bank and fills in the entry
// in the bankLimits vector.
static void fillBankLimits(int numBanks, int64_t bankSize,
                           std::vector<BankLimits> &bankLimits) {
  for (int i = 0; i < numBanks; i++) {
    auto startAddr = bankSize * i;
    auto endAddr = bankSize * (i + 1);
    bankLimits.push_back({startAddr, endAddr});
  }
}

namespace {
// Byte-granular map of which bytes of one tile's data memory are taken.
// Replaces the old per-bank "next free address" watermark -- a bump pointer
// that cannot represent a hole, so a fixed-address buffer stranded every free
// byte below it -- at a cost of at most a 64 kB bitmap (a tile is at most
// 512 kB). Byte, not bus-width, granularity: buffers marked `aligned = false`
// are packed at unaligned offsets on purpose.
class MemoryOccupancy {
public:
  explicit MemoryOccupancy(int64_t size) : occupied(size, false) {}

  int64_t size() const { return occupied.size(); }

  // True when [start, end) lies inside the tile and no byte of it is taken.
  bool isRangeFree(int64_t start, int64_t end) const {
    if (start < 0 || end > size() || start > end)
      return false;
    return start == end || occupied.find_first_in(start, end) == -1;
  }

  void markOccupied(int64_t start, int64_t end) {
    assert(start >= 0 && end <= size() && start <= end &&
           "range must lie inside the tile");
    if (start < end)
      occupied.set(start, end);
  }

  // Undo a markOccupied. Only used to lift the core's data-region placeholder
  // once every buffer is placed: nothing can have been placed inside it while
  // it was marked, so clearing exactly its range is exact rather than
  // approximate.
  void markFree(int64_t start, int64_t end) {
    assert(start >= 0 && end <= size() && start <= end &&
           "range must lie inside the tile");
    if (start < end)
      occupied.reset(start, end);
  }

  // Placement for `size` bytes in [lo, hi) that leaves the largest single free
  // run behind anywhere in [0, size()), or nullopt when nothing fits.
  //
  // For an object placed *before* the unconstrained buffers -- the core's data
  // region, or a bank-pinned buffer -- where it sits inside the space it is
  // allowed to use is not arbitrary. Dropped into the middle of a hole it
  // bisects the free space, and a later buffer needing one long run has
  // nowhere to go; slid flush against something already occupied it takes the
  // same bytes and leaves the remainder whole. Both such objects are
  // indifferent to which end they land on, so pick by what survives.
  //
  // Only the two flush positions of each candidate hole are considered, since
  // any placement strictly inside a hole leaves strictly less contiguous space
  // than flushing to one of its ends.
  std::optional<int64_t> findLeastFragmentingGap(int64_t lo, int64_t hi,
                                                 int64_t size,
                                                 int64_t alignBytes) const {
    assert(alignBytes > 0 && "alignment must be positive");
    std::optional<int64_t> best;
    int64_t bestRun = -1;
    auto consider = [&](int64_t start) {
      MemoryOccupancy trial = *this;
      trial.markOccupied(start, start + size);
      int64_t run = trial.largestGap(0, this->size());
      // Ties go to the lowest address, so placement stays deterministic.
      if (run > bestRun || (run == bestRun && best && start < *best)) {
        bestRun = run;
        best = start;
      }
    };
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      int64_t low = llvm::alignTo(gapStart, alignBytes);
      if (low + size > gapEnd)
        return;
      consider(low);
      // Flush to the top, then back down to the alignment; that can only move
      // the start lower, and never below `low`.
      consider(
          std::max<int64_t>(llvm::alignDown(gapEnd - size, alignBytes), low));
    });
    return best;
  }

  // Start of the tightest gap in [lo, hi) that holds `size` bytes, or nullopt.
  // Ties go to the lowest address, so placement is deterministic. Candidate
  // starts are aligned up *before* the fit test, so a gap is never rejected
  // just because its first free byte is misaligned.
  std::optional<int64_t> findGap(int64_t lo, int64_t hi, int64_t size,
                                 int64_t alignBytes) const {
    assert(alignBytes > 0 && "alignment must be positive");
    std::optional<int64_t> best;
    int64_t bestSlack = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      int64_t start = llvm::alignTo(gapStart, alignBytes);
      if (start + size <= gapEnd) {
        // Waste is measured across the whole gap, so a hole that would lose a
        // lot of its front to alignment padding is not mistaken for a tight
        // fit and preferred over one that genuinely wastes less.
        int64_t slack = (gapEnd - gapStart) - size;
        if (!best || slack < bestSlack) {
          best = start;
          bestSlack = slack;
        }
      }
    });
    return best;
  }

  // Total free bytes in [lo, hi), for diagnostics that need to say how far
  // off a failed placement was, not just that it failed.
  int64_t freeBytes(int64_t lo, int64_t hi) const {
    int64_t total = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      total += gapEnd - gapStart;
    });
    return total;
  }

  // Bytes of the enclosing free run that `size` bytes at `addr` would leave
  // unused. Placement scoring uses this to prefer a tight fit, so that large
  // clean regions stay available for buffers that need them.
  int64_t slackAt(int64_t addr, int64_t size) const {
    int64_t slack = 0;
    forEachGap(0, this->size(), [&](int64_t gapStart, int64_t gapEnd) {
      if (addr >= gapStart && addr + size <= gapEnd)
        slack = (gapEnd - gapStart) - size;
    });
    return slack;
  }

  // Largest single free run in [lo, hi). Distinct from freeBytes: a
  // reservation needs one contiguous run, so the total says nothing useful
  // about whether it fits.
  int64_t largestGap(int64_t lo, int64_t hi) const {
    int64_t best = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      best = std::max(best, gapEnd - gapStart);
    });
    return best;
  }

private:
  // Calls fn(gapStart, gapEnd) for every maximal free run in [lo, hi).
  // Shared by findGap and freeBytes so the two can't drift on what counts as
  // a gap.
  template <typename Fn>
  void forEachGap(int64_t lo, int64_t hi, Fn fn) const {
    lo = std::max<int64_t>(lo, 0);
    hi = std::min(hi, size());
    for (int64_t cursor = lo; cursor < hi;) {
      int gapStart = occupied.find_first_unset_in(cursor, hi);
      if (gapStart == -1)
        break;
      int nextTaken = occupied.find_first_in(gapStart, hi);
      // find_first_in cannot return gapStart (it is clear), so gapEnd >
      // gapStart >= cursor and the cursor always advances.
      int64_t gapEnd = nextTaken == -1 ? hi : nextTaken;
      fn(gapStart, gapEnd);
      cursor = gapEnd;
    }
  }

  llvm::BitVector occupied;
};
} // namespace

// Alignment a buffer must be placed at, in bytes. Buffers marked
// `aligned = false` may start anywhere.
static int64_t getBufferAlignBytes(BufferOp buffer,
                                   uint32_t tileAlignBitWidth) {
  if (!buffer.getAligned())
    return 1;
  return std::max<int64_t>(tileAlignBitWidth / 8, 1);
}

// Index of the bank owning `addr`, or -1 when it falls outside every bank.
static int getBankContaining(int64_t addr, int numBanks,
                             ArrayRef<BankLimits> bankLimits) {
  for (int i = 0; i < numBanks; i++)
    if (addr >= bankLimits[i].startAddr && addr < bankLimits[i].endAddr)
      return i;
  return -1;
}

// Sets the buffer's address and mem_bank attributes and marks the bytes it
// covers as taken.
static void placeBuffer(BufferOp buffer, int64_t startAddr, int bank,
                        MemoryOccupancy &occupancy) {
  buffer.setAddress(startAddr);
  buffer.setMemBank(bank);
  occupancy.markOccupied(startAddr, startAddr + buffer.getAllocationSize());
}

// The tile-level constants every bank-aware helper below needs to agree on.
// Bundled so a helper's parameter list describes what actually varies from
// call to call (the buffer, the occupancy, the strategy) instead of each one
// re-threading the same six invariants down from simpleBankAwareAllocation.
struct BankAwareContext {
  int numBanks;
  uint32_t tileAlignBitWidth;
  ArrayRef<BankLimits> bankLimits;
  int64_t maxDataMemorySize;
  int64_t stacksize;
  int64_t reservedData;
};

// Places a buffer carrying an explicit `address`, checking that the space it
// asks for is free and that any `mem_bank` it also carries agrees. Returns
// false when the buffer has no address at all, leaving it to the mem_bank or
// free-placement path; returns failure when the address is unusable -- every
// such failure has already emitted an error, so the caller must treat it as
// terminal rather than falling back to another scheme (see BankAwareResult
// below for why).
static FailureOr<bool>
checkAndAddBufferWithAddress(BufferOp buffer, const BankAwareContext &ctx,
                             MemoryOccupancy &occupancy) {
  auto addrOpt = buffer.getAddress();
  if (!addrOpt)
    return false;
  // it is fine if mem_bank is not set
  auto memBankOpt = buffer.getMemBank();

  int64_t addr = *addrOpt;
  if (buffer.getAligned() &&
      addr % getBufferAlignBytes(buffer, ctx.tileAlignBitWidth) != 0) {
    return buffer->emitOpError(
        "address attribute value must be aligned to tile load/store bus width "
        "when aligned attribute is set");
  }

  int bank = getBankContaining(addr, ctx.numBanks, ctx.bankLimits);
  if (bank < 0) {
    // A zero-sized buffer covers no bytes, so pinning it exactly at the top
    // of the tile's memory -- one past every bank's range -- is legal: treat
    // it as belonging to the last bank, since there is nothing there for it
    // to conflict with.
    if (buffer.getAllocationSize() == 0 &&
        addr == ctx.bankLimits.back().endAddr)
      bank = ctx.numBanks - 1;
    else
      return buffer->emitOpError(
          "address attribute does not fall within any bank range");
  }

  int64_t endAddr = addr + buffer.getAllocationSize();
  if (endAddr > occupancy.size())
    return buffer->emitOpError("address attribute would place the buffer past "
                               "the end of the tile's memory");

  // A pinned address is only invalid when it actually collides with something
  // already placed. Note the buffer is deliberately *not* required to stay
  // inside `bank`: the hardware places no natural-size or bank alignment
  // requirement on a buffer, so a pinned buffer may straddle a bank boundary.
  if (!occupancy.isRangeFree(addr, endAddr))
    return buffer->emitOpError("would override allocated address");

  if (memBankOpt && *memBankOpt != bank)
    return buffer->emitOpError(
        "mem_bank attribute is inconsistent with address attribute");
  placeBuffer(buffer, addr, bank, occupancy);
  return true;
}

// Bank a buffer is required to live in because the user asked for it. This is
// recorded separately from the `mem_bank` attribute because the allocator also
// writes that attribute for buffers it places itself, and clears it when
// rolling back a failed attempt.
using RequiredBanks = DenseMap<Operation *, int>;

// A buffer carrying only a `mem_bank` still has to be given an address, so it
// is placed with the other unplaced buffers rather than ahead of them; this
// just records the request and rejects a bank that does not exist.
static LogicalResult recordRequiredBank(BufferOp buffer, int numBanks,
                                        RequiredBanks &requiredBanks) {
  auto memBankOpt = buffer.getMemBank();
  if (!memBankOpt)
    return success();
  if (*memBankOpt < 0 || *memBankOpt >= numBanks)
    return buffer->emitOpError("mem_bank attribute value is out of range");
  requiredBanks[buffer] = *memBankOpt;
  return success();
}

// Prints the memory map across banks
static void printMemMap(TileOp tile, ArrayRef<BufferOp> allocatedBuffers,
                        ArrayRef<BufferOp> preAllocatedBuffers,
                        const BankAwareContext &ctx) {
  InFlightDiagnostic error = tile.emitWarning(
      "Not all requested buffers fit in the available memory.\n");
  auto &note = error.attachNote()
               << "Current configuration of buffers in bank(s) : ";
  note << "MemoryMap:\n";
  auto printbuffer = [&](StringRef name, int64_t address, int64_t size,
                         StringRef suffix = "") {
    printMemoryMapEntry(note, name, address, size, /*indent=*/2, suffix);
  };
  for (int i = 0; i < ctx.numBanks; i++) {
    if (i == 0) {
      if (ctx.stacksize > 0)
        printbuffer("(stack)", 0, ctx.stacksize);
      else
        note << "\t(no stack allocated)\n";
      if (auto computed = getComputedStackRequirement(tile))
        note << "\t(call-graph analysis computed this core's callees need >= "
             << *computed << " bytes of stack)\n";
    }
    note << "\t"
         << "bank : " << i << "\t"
         << "0x" << llvm::utohexstr(ctx.bankLimits[i].startAddr) << "-0x"
         << llvm::utohexstr(ctx.bankLimits[i].endAddr - 1) << "\n";
    // This runs on the failure path, so some buffers have no address yet.
    auto printPlaced = [&](ArrayRef<BufferOp> buffers) {
      for (auto buffer : buffers) {
        auto addrOpt = buffer.getAddress();
        auto memBankOpt = buffer.getMemBank();
        if (!addrOpt || !memBankOpt || *memBankOpt != i)
          continue;
        int64_t size = buffer.getAllocationSize();
        // Listed under its start bank only (mem_bank records one bank), but a
        // buffer too big for one bank straddles rather than failing, so its
        // range can genuinely run past the bank printed above it.
        std::string suffix;
        if (*addrOpt + size > ctx.bankLimits[i].endAddr)
          suffix = (" (straddles into bank " + llvm::Twine(i + 1) + ")").str();
        printbuffer(buffer.name(), *addrOpt, size, suffix);
      }
    };
    printPlaced(preAllocatedBuffers);
    printPlaced(allocatedBuffers);
  }
}

// Places a buffer in the first bank, starting round-robin from the given
// index, that has a hole big enough, taking the tightest such hole so that
// large clean regions stay available for large buffers. Spreading over banks
// only limits DMA contention, not buffer size, so one that fits no single
// bank straddles bank boundaries instead of being rejected; `preferBankAligned`
// starts a straddling buffer on a bank boundary (fewest banks touched, but can
// strand up to a bank of space ahead of it), and the caller retries without it
// rather than reporting the tile full. Returns false if there is no room for
// the buffer at all; the caller reports which buffer failed.
static bool setBufferAddress(BufferOp buffer, const BankAwareContext &ctx,
                             int &startBankIndex,
                             const RequiredBanks &requiredBanks,
                             MemoryOccupancy &occupancy) {
  assert(startBankIndex < ctx.numBanks &&
         "Unexpected input value for startBankIndex");
  int64_t size = buffer.getAllocationSize();
  int64_t alignBytes = getBufferAlignBytes(buffer, ctx.tileAlignBitWidth);

  auto place = [&](int64_t startAddr, int bank) {
    placeBuffer(buffer, startAddr, bank, occupancy);
    startBankIndex = (bank + 1) % ctx.numBanks;
    return true;
  };

  // A requested mem_bank is a hard constraint: no other bank, and no
  // straddling. It may use any hole inside its own bank. It also leaves the
  // round-robin cursor alone, since the user chose this bank, not the spread.
  auto required = requiredBanks.find(buffer);
  if (required != requiredBanks.end()) {
    int bank = required->second;
    // Where inside the bank matters, because this buffer is placed before the
    // unconstrained ones. A small pinned buffer dropped into the middle of the
    // free space bisects it, and a large buffer that needs a run spanning
    // several banks then has nowhere to go -- even though sliding the pinned
    // one to the end of its bank would have left that run intact. Its own
    // constraint is satisfied anywhere in the bank, so pick the spot that
    // leaves the largest single run behind.
    if (auto startAddr = occupancy.findLeastFragmentingGap(
            ctx.bankLimits[bank].startAddr, ctx.bankLimits[bank].endAddr, size,
            alignBytes)) {
      placeBuffer(buffer, *startAddr, bank, occupancy);
      return true;
    }
    // A zero-sized buffer covers no bytes, so no bank can be too full to hold
    // it; findGap only failed for want of a free byte it will never use.
    if (size == 0) {
      placeBuffer(buffer, ctx.bankLimits[bank].startAddr, bank, occupancy);
      return true;
    }
    return false;
  }

  // Score every placement this buffer could take and keep the best, rather
  // than trying one rule and falling back to another. The keys, in order:
  //
  //  1. banks touched, fewest first. Spanning a bank boundary costs DMA
  //     bandwidth on every bank it lands in, so a buffer that fits one bank
  //     should stay in one.
  //  2. round-robin distance from the cursor, nearest first. Spreading over
  //     banks is what limits DMA contention. Every single-bank placement ties
  //     on key 1, so this only ever decides between placements that already
  //     fit one bank -- exactly the old round-robin scan.
  //  3. slack, tightest first, so large clean regions stay available.
  //  4. address, lowest first, purely so the result is deterministic.
  //
  // Key 1 is what the old `preferBankAligned` was reaching for: it started a
  // straddling buffer on a bank boundary in the hope of touching fewer banks,
  // and the caller retried without it when that stranded too much. Measuring
  // banks-touched directly makes a boundary start win only when it actually
  // reduces them, so the retry folds in here.
  int64_t bankedEnd = ctx.bankLimits.back().endAddr;
  int64_t bankSize = bankedEnd / ctx.numBanks;
  struct Candidate {
    int64_t addr;
    int bank;
    int64_t banksTouched;
    int64_t rrDistance;
    int64_t slack;
  };
  std::optional<Candidate> best;
  auto consider = [&](std::optional<int64_t> addr) {
    if (!addr)
      return;
    int bank = getBankContaining(*addr, ctx.numBanks, ctx.bankLimits);
    if (bank < 0)
      return;
    int64_t touched =
        size == 0 ? 1 : (*addr + size - 1) / bankSize - *addr / bankSize + 1;
    Candidate c{*addr, bank, touched,
                (bank - startBankIndex + ctx.numBanks) % ctx.numBanks,
                occupancy.slackAt(*addr, size)};
    auto rank = [](const Candidate &x) {
      return std::tie(x.banksTouched, x.rrDistance, x.slack, x.addr);
    };
    if (!best || rank(c) < rank(*best))
      best = c;
  };

  for (int i = 0; i < ctx.numBanks; i++)
    consider(occupancy.findGap(ctx.bankLimits[i].startAddr,
                               ctx.bankLimits[i].endAddr, size, alignBytes));
  // Nothing may fit a single bank; straddling is better than giving up.
  // Searched over the banked region only, so the result always maps to a bank.
  consider(occupancy.findGap(0, bankedEnd, size, alignBytes));
  // Only ask for bank-boundary starts when a bank boundary actually satisfies
  // the buffer's own alignment; otherwise this would silently search on a
  // stride that is neither a bank boundary nor what was intended.
  if (bankSize % alignBytes == 0)
    consider(occupancy.findGap(0, bankedEnd, size, bankSize));

  if (best)
    return place(best->addr, best->bank);

  // A zero-sized buffer covers no bytes, so a tile with no hole left in it can
  // still hold one. It is excluded from the overlap checks for the same
  // reason.
  if (size == 0)
    return place(0, 0);

  return false;
}

// Places every buffer in `buffersToAlloc`, in order, into the free space left
// by the pre-allocated ones. Returns the first buffer that did not fit, or
// nullptr when they all did; `placed` collects what was assigned so a failed
// attempt can be rolled back.
static BufferOp placeFreeBuffers(ArrayRef<BufferOp> buffersToAlloc,
                                 const BankAwareContext &ctx,
                                 const RequiredBanks &requiredBanks,
                                 MemoryOccupancy &occupancy,
                                 SmallVectorImpl<BufferOp> &placed) {
  int startBankIndex = 0;
  for (auto buffer : buffersToAlloc) {
    if (!setBufferAddress(buffer, ctx, startBankIndex, requiredBanks,
                          occupancy))
      return buffer;
    placed.push_back(buffer);
  }
  return nullptr;
}

// Rolls back what the allocator wrote, leaving a mem_bank the user asked for
// in place.
static void deAllocationBuffers(SmallVectorImpl<BufferOp> &buffers,
                                const RequiredBanks &requiredBanks) {
  for (auto buffer : buffers) {
    buffer->removeAttr("address");
    if (!requiredBanks.count(buffer))
      buffer->removeAttr("mem_bank");
  }
}

// Why bank-aware allocation stopped. A design that simply does not fit may be
// worth retrying with another scheme; a constraint the user wrote and that
// cannot be honoured must not be, because the other scheme ignores mem_bank
// and would silently place the buffer in a different bank than was asked for.
namespace {
enum class BankAwareResult { Success, OutOfMemory, ConstraintUnsatisfiable };
} // namespace

// Places every buffer carrying an explicit `address`, address-first so a
// mem_bank-only buffer can't carve up space an address pin needs. Anything
// left with only a `mem_bank` is recorded as required and queued into
// `buffersToAlloc` for the strategy portfolio to place. Failure here is
// always terminal: an error has already been emitted, and retrying under
// another scheme would either hit the same problem or silently drop the
// constraint (see BankAwareResult below for why that isn't acceptable).
static LogicalResult placePreAllocatedBuffers(
    SmallVectorImpl<BufferOp> &preAllocatedBuffers, const BankAwareContext &ctx,
    MemoryOccupancy &occupancy, RequiredBanks &requiredBanks,
    SmallVectorImpl<BufferOp> &buffersToAlloc) {
  // Address buffers first (ascending, within the same bank), then
  // mem_bank-only buffers; otherwise stable.
  std::sort(preAllocatedBuffers.begin(), preAllocatedBuffers.end(),
            [](BufferOp a, BufferOp b) -> bool {
              auto a_addr = a.getAddress();
              auto b_addr = b.getAddress();
              if (a_addr.has_value() && b_addr.has_value())
                return a_addr.value() < b_addr.value();
              return a_addr.has_value() && !b_addr.has_value();
            });

  for (auto buffer : preAllocatedBuffers) {
    auto has_addr = checkAndAddBufferWithAddress(buffer, ctx, occupancy);
    if (failed(has_addr))
      return failure();
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (*has_addr)
      continue;
    // Only a mem_bank: the address is still ours to choose, so queue it with
    // the rest instead of letting a small constrained buffer carve up the free
    // space before the large buffers have had a chance at it.
    if (failed(recordRequiredBank(buffer, ctx.numBanks, requiredBanks)))
      return failure();
    buffersToAlloc.push_back(buffer);
  }
  return success();
}

// Order buffers for placement: bank-pinned first, then largest first.
//
// Bank-pinned first is the standard most-constrained-variable rule -- a
// `mem_bank` request has exactly one candidate bank, so it has the least room
// to move and the least chance of finding a home later. Largest first among
// the rest keeps small buffers from fragmenting the clean regions the large
// ones need.
static SmallVector<BufferOp>
placementOrder(ArrayRef<BufferOp> buffersToAlloc,
               const RequiredBanks &requiredBanks) {
  SmallVector<BufferOp> order(buffersToAlloc.begin(), buffersToAlloc.end());
  llvm::stable_sort(order, [&](BufferOp a, BufferOp b) {
    if (requiredBanks.count(a) != requiredBanks.count(b))
      return requiredBanks.count(a) > requiredBanks.count(b);
    return a.getAllocationSize() > b.getAllocationSize();
  });
  return order;
}

// Reserve `reservedData` contiguous bytes for the core's own sections, as a
// placeholder the free buffers cannot then place into.
//
// This runs *after* the bank-pinned buffers and before the unconstrained ones.
// The order is deliberate: a region placed before the pins could take the only
// space a `mem_bank` request -- a hard constraint the user wrote -- had left,
// turning a working design into a constraint error. Pins first can only make
// the region's own search harder, never make a hard constraint unsatisfiable.
//
// Returns the placeholder's range, or nullopt when nothing large enough is
// free, in which case a diagnostic has already been emitted.
static std::optional<MemoryRun>
reserveCoreDataRegion(TileOp tile, const BankAwareContext &ctx,
                      MemoryOccupancy &occupancy) {
  if (ctx.reservedData <= 0)
    return MemoryRun{0, 0};
  // Byte alignment: the linker script's ORIGIN has always been an arbitrary
  // byte, and asking for more here only shrinks the set of placements that fit.
  if (auto start = occupancy.findLeastFragmentingGap(
          0, ctx.maxDataMemorySize, ctx.reservedData, /*alignBytes=*/1)) {
    occupancy.markOccupied(*start, *start + ctx.reservedData);
    return MemoryRun{*start, ctx.reservedData};
  }

  // Emitted before any free buffer is placed, so the only things in the way are
  // the stack and the user's own pins -- a short, actionable list rather than
  // "some buffer somewhere did not fit".
  tile.emitOpError("cannot reserve ")
      << ctx.reservedData
      << " contiguous bytes for this core's data sections "
         "(reserved_data_size); the largest free run is "
      << occupancy.largestGap(0, ctx.maxDataMemorySize)
      << " bytes. Only the stack and this tile's address- or bank-pinned "
         "buffers are placed at this point, so it is one of those, or the "
         "reservation itself, that has to give";
  return std::nullopt;
}

static BankAwareResult simpleBankAwareAllocation(TileOp tile) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device)
    return BankAwareResult::OutOfMemory;

  clearCoreDataStamp(tile);

  std::vector<BankLimits> bankLimits; // the entries contain pairs of start and
                                      // end addresses for each bank

  const auto &targetModel = getTargetModel(tile);
  auto [maxDataMemorySize, tileAlignBitWidth] =
      tileMemoryLimits(tile, targetModel);

  int numBanks = targetModel.getNumBanks(tile.getCol(), tile.getRow());
  int64_t bankSize = maxDataMemorySize / numBanks;

  // Address range owned by the MemTile is 0x80000.
  // Address range owned by the tile is 0x8000 in
  // AIE1 and 0x10000 in AIE2, but we need room at
  // the bottom for stack.
  int64_t stacksize = 0;
  int64_t reservedData = 0;
  MemoryOccupancy occupancy(maxDataMemorySize);
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getEffectiveStackSize();
    occupancy.markOccupied(0, std::min<int64_t>(stacksize, maxDataMemorySize));
    // Space the core's own compiled sections need; see reserved_data_size.
    reservedData = core.getReservedDataSize().value_or(0);
  }
  fillBankLimits(numBanks, bankSize, bankLimits);
  BankAwareContext ctx{numBanks,          tileAlignBitWidth, bankLimits,
                       maxDataMemorySize, stacksize,         reservedData};

  RequiredBanks requiredBanks;
  SmallVector<BufferOp> preAllocatedBuffers;
  SmallVector<BufferOp> buffersToAlloc;
  SmallVector<BufferOp> allBuffers_on_tile;
  // Collect all the buffers for this tile.
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile) {
      if (!isBufferPreAllocated(buffer)) {
        buffersToAlloc.push_back(buffer);
      } else {
        preAllocatedBuffers.push_back(buffer);
      }
      allBuffers_on_tile.push_back(buffer);
    }
  });

  if (failed(placePreAllocatedBuffers(preAllocatedBuffers, ctx, occupancy,
                                      requiredBanks, buffersToAlloc)))
    return BankAwareResult::ConstraintUnsatisfiable;

  // Keeps track of buffers this pass placed, to be able to deallocate in case
  // of failure and print helpful debug info about them. This does not include
  // the pre-allocated buffers.
  SmallVector<BufferOp> allocatedBuffers;

  // One pass, in three steps. Buffers are ordered once (most-constrained
  // first, then largest first); the bank-pinned prefix goes down, then the
  // core's data region is placed into what is left, then the unconstrained
  // buffers fill in around it.
  //
  // The region is placed rather than measured afterwards, which is what lets
  // this be a single pass at all: the old code could only check the leftover
  // run once everything was down, so a shortfall meant re-placing everything
  // in a different order and checking again.
  SmallVector<BufferOp> order = placementOrder(buffersToAlloc, requiredBanks);
  auto pinnedEnd = llvm::partition_point(
      order, [&](BufferOp b) { return requiredBanks.count(b) > 0; });
  ArrayRef<BufferOp> pinnedPrefix(order.begin(), pinnedEnd);
  ArrayRef<BufferOp> freeSuffix(pinnedEnd, order.end());

  BufferOp failedBuffer = placeFreeBuffers(pinnedPrefix, ctx, requiredBanks,
                                           occupancy, allocatedBuffers);

  std::optional<MemoryRun> dataPlaceholder;
  if (!failedBuffer) {
    dataPlaceholder = reserveCoreDataRegion(tile, ctx, occupancy);
    if (!dataPlaceholder) {
      // reserveCoreDataRegion has already reported which obstacle is in the
      // way. Fall through to basic-sequential the way any other out-of-room
      // outcome does; it packs from the bottom and may still leave the run.
      deAllocationBuffers(allocatedBuffers, requiredBanks);
      return BankAwareResult::OutOfMemory;
    }
    failedBuffer = placeFreeBuffers(freeSuffix, ctx, requiredBanks, occupancy,
                                    allocatedBuffers);
  }

  if (BufferOp failed = failedBuffer) {
    // A buffer pinned to a bank that cannot hold it is a constraint the user
    // wrote, not a tile that ran out of room, so it gets its own error and no
    // memory map.
    if (requiredBanks.count(failed)) {
      int bank = requiredBanks.lookup(failed);
      int64_t need = failed.getAllocationSize();
      int64_t bankCapacity =
          bankLimits[bank].endAddr - bankLimits[bank].startAddr;
      if (need > bankCapacity)
        failed->emitOpError("requires ")
            << need << " bytes, which cannot fit in bank " << bank << " ("
            << bankCapacity << " bytes total)";
      else
        failed->emitOpError("requires ")
            << need << " bytes in bank " << bank << ", but only "
            << occupancy.freeBytes(bankLimits[bank].startAddr,
                                   bankLimits[bank].endAddr)
            << " of " << bankCapacity << " bytes are free there";
      deAllocationBuffers(allocatedBuffers, requiredBanks);
      return BankAwareResult::ConstraintUnsatisfiable;
    }
    failed.emitWarning("Failed to allocate buffer: ")
        << failed.name() << " with size: " << failed.getAllocationSize()
        << " bytes.";
    // The reservation is invisible in the memory map -- it is a placeholder in
    // the bitmap, not a buffer -- so without this a user sees only that some
    // buffer did not fit, with no hint that a reservation took the space. Say
    // whether lifting it would have been enough, so the next step is either
    // "shrink reserved_data_size" or "this tile is genuinely full".
    if (dataPlaceholder && dataPlaceholder->size) {
      MemoryOccupancy without = occupancy;
      without.markFree(dataPlaceholder->start,
                       dataPlaceholder->start + dataPlaceholder->size);
      int64_t need = failed.getAllocationSize();
      bool wouldFit =
          without.largestGap(0, ctx.maxDataMemorySize) >= need || need == 0;
      failed.emitRemark("this core reserves ")
          << dataPlaceholder->size
          << " bytes for its own data sections (reserved_data_size), placed at "
             "0x"
          << llvm::utohexstr(dataPlaceholder->start) << "; '"
          << failed.name().getValue() << "' "
          << (wouldFit ? "would have fit without that reservation"
                       : "would not have fit even without that reservation");
    }
    // The memory map reads the addresses handed out, so print before rolling
    // them back.
    printMemMap(tile, allocatedBuffers, preAllocatedBuffers, ctx);
    deAllocationBuffers(allocatedBuffers, requiredBanks);
    return BankAwareResult::OutOfMemory;
  }
  assert(allocatedBuffers.size() == buffersToAlloc.size());

  // Sort by smallest address before printing memory map.
  sortBuffersByAddress(allBuffers_on_tile);
  // Every placement above was taken from free space inside the tile, so a
  // bank/tile overflow is no longer representable here; the stack and overlap
  // checks remain as a backstop over the final addresses.
  if (!checkAndPrintOverlapStackframe(stacksize, allBuffers_on_tile) ||
      !checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth))
    return BankAwareResult::OutOfMemory;
  // The placeholder has done its job: it kept the free buffers out of the
  // region while they were being placed. Lift it and stamp the run that
  // actually survived, which has grown to absorb whatever the buffers did not
  // use -- so the core still receives all the leftover memory, and the grant
  // is >= the request rather than equal to it.
  if (dataPlaceholder && dataPlaceholder->size)
    occupancy.markFree(dataPlaceholder->start,
                       dataPlaceholder->start + dataPlaceholder->size);
  stampCoreDataRegion(
      tile, coreDataRun(maxDataMemorySize, stacksize, allBuffers_on_tile));
  return BankAwareResult::Success;
}

static LogicalResult checkBufferScope(BufferOp buffer, DeviceOp device) {
  // Buffers are not allowed to be inside the core without being statically
  // initialized.
  Operation *parent = buffer->getParentOp();
  // Allowed to be in MemTile
  if (!isa<DeviceOp>(parent) && !isa<MemTileDMAOp>(parent) &&
      !buffer.getInitialValue().has_value()) {
    auto tile = buffer.getTileOp();
    tile->emitOpError("Buffer '")
        << buffer.name()
        << "' must be defined directly under the device scope. Currently it "
           "is nested inside a core tile.";
    return failure();
  }
  return success();
}

namespace {
struct AIEAssignBufferAddressesPass
    : xilinx::AIE::impl::AIEAssignBufferAddressesBase<
          AIEAssignBufferAddressesPass> {

  AIEAssignBufferAddressesPass() = default;

  AIEAssignBufferAddressesPass(const AIEAssignBufferAddressesOptions &options) {
    clAllocScheme = options.clAllocScheme;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    // Ensure all BufferOps are globally defined at the device level.
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (failed(checkBufferScope(buffer, device)))
        return signalPassFailure();
    });
    // Make sure all the buffers have a name
    int counter = 0;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (!buffer.hasName()) {
        std::string name = "_anonymous";
        name += std::to_string(counter++);
        buffer->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      }
    });

    // Select allocation scheme per tile
    for (auto tile : device.getOps<TileOp>()) {
      auto tileAllocationScheme = tile.getAllocationScheme();

      if (!tileAllocationScheme)
        tileAllocationScheme = clAllocScheme;

      if (tileAllocationScheme == "basic-sequential") {
        if (!basicAllocation(tile)) {
          tile.emitOpError("Basic sequential allocation failed.");
          return signalPassFailure();
        }
      } else if (tileAllocationScheme == "bank-aware") {
        if (simpleBankAwareAllocation(tile) != BankAwareResult::Success) {
          tile.emitOpError("Bank-aware allocation failed.");
          return signalPassFailure();
        }
      } else {
        switch (simpleBankAwareAllocation(tile)) {
        case BankAwareResult::Success:
          break;
        case BankAwareResult::ConstraintUnsatisfiable:
          // See BankAwareResult's definition for why this can't retry under
          // basic-sequential. Report the constraint instead.
          tile.emitOpError("Bank-aware allocation failed.");
          return signalPassFailure();
        case BankAwareResult::OutOfMemory: {
          // basic-sequential has no notion of banks, so a buffer that was
          // only ever going to get an address from bank-aware placement
          // (mem_bank but no address) would silently lose that guarantee --
          // even though its own pin may have been perfectly satisfiable and
          // bank-aware failed for an unrelated reason (another buffer, or
          // reserved_data_size). Only an address pin is safe to retry under
          // basic-sequential, since that scheme does honour it.
          SmallVector<BufferOp> droppedPins;
          device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
            if (buffer.getTileOp() == tile && buffer.getMemBank() &&
                !buffer.getAddress())
              droppedPins.push_back(buffer);
          });
          if (!droppedPins.empty()) {
            InFlightDiagnostic diag = tile.emitOpError(
                "bank-aware allocation failed; falling back to "
                "basic-sequential would silently drop the mem_bank pin on: ");
            for (auto [i, buffer] : llvm::enumerate(droppedPins)) {
              if (i)
                diag << ", ";
              diag << buffer.name();
            }
            return signalPassFailure();
          }
          tile.emitWarning("Bank-aware allocation failed, trying basic "
                           "sequential allocation.");
          if (!basicAllocation(tile)) {
            tile.emitOpError("Basic sequential allocation also failed.");
            return signalPassFailure();
          }
          break;
        }
        }
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass() {
  return std::make_unique<AIEAssignBufferAddressesPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass(
    const AIEAssignBufferAddressesOptions &options) {
  return std::make_unique<AIEAssignBufferAddressesPass>(options);
}
