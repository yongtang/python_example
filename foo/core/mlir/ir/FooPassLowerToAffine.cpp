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

struct FooLoweringToAffinePass
    : public PassWrapper<FooLoweringToAffinePass, FunctionPass> {
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
    // target.addLegalOp<foo::TODO>();

    // Provide the set of patterns that will lower the Foo operations.
    OwningRewritePatternList patterns;
    // patterns.insert<TODO>(&getContext());

    // Signal failure if any `illegal` operations were not converted
    // successfully.
    if (failed(applyPartialConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createLowerToAffinePass() {
  return std::make_unique<FooLoweringToAffinePass>();
}
}  // namespace foo
}  // namespace mlir
