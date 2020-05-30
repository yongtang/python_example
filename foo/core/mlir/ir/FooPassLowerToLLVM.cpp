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
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace foo {
namespace {

struct FooToLLVMLoweringPass
    : public PassWrapper<FooToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() final {
    // Only targeting LLVM dialect.
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    // Lowering the MemRef types being operated on, to a representation in LLVM.
    LLVMTypeConverter typeConverter(&getContext());

    // Provide the patterns used for lowering.
    OwningRewritePatternList patterns;
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    // Remaining operation to lower
    // patterns.insert<TODO>(&getContext());

    // Use a `FullConversion` to ensure only legal operations will remain.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, patterns, &typeConverter))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<FooToLLVMLoweringPass>();
}
}  // namespace foo
}  // namespace mlir
