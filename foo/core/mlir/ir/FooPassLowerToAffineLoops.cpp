/* Copyright 2020 Yong Tang. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "foo/core/mlir/ir/FooDialect.h"
#include "foo/core/mlir/ir/FooOps.h"
#include "foo/core/mlir/ir/FooPasses.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace foo {
namespace {
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block.
  // This is fine as foo functions have no control flow.
  auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

struct ConstantOpLowering : public OpRewritePattern<foo::ConstantOp> {
  using OpRewritePattern<foo::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(foo::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.value();
    Location loc = op.getLoc();

    // Allocate and assign constant values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Generating constant indices up-to the largest dimension.
    // Create up-front to avoid large amounts of redundant operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end()))) {
        constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
      }
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    }

    // A multi-dimensional constant, generate a store for each of the elements.
    // Recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.getValues<FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion,
      // store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<mlir::ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension,
      //  and add the indices to the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<foo::ReturnOp> {
  using OpRewritePattern<foo::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(foo::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // All function calls should have been inlined.
    if (op.hasOperand()) {
      return failure();
    }
    // Lower "foo.return" directly to "std.return".
    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
    return success();
  }
};

struct FooToAffineLoweringPass
    : public PassWrapper<FooToAffineLoweringPass, FunctionPass> {
  void runOnFunction() final {
    auto function = getFunction();

    // Only lower the main function as all other functions have been inlined.
    if (function.getName() != "main") {
      return;
    }

    // Verify no inputs and results.
    if (function.getNumArguments() || function.getType().getNumResults()) {
      function.emitError("expected 'main' to have 0 inputs and 0 results");
      return signalPassFailure();
    }

    // Define the conversion target.
    ConversionTarget target(getContext());

    // Define the dialects that are legal targets.
    target.addLegalDialect<AffineDialect, StandardOpsDialect>();

    // Define the Foo dialect as Illegal, so all operatsions are converted.
    // Explicitly mark the Foo operations, `foo.print`, as `legal`.
    target.addIllegalDialect<foo::FooDialect>();
    target.addLegalOp<foo::PrintOp>();

    // Provide the set of patterns that will lower the Foo operations.
    OwningRewritePatternList patterns;
    patterns.insert<ConstantOpLowering, ReturnOpLowering>(&getContext());

    // Signal failure if any `illegal` operations were not converted
    // successfully.
    if (failed(applyPartialConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createLowerToAffinePass() {
  return std::make_unique<FooToAffineLoweringPass>();
}
}  // namespace foo
}  // namespace mlir
