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
#include "foo/core/mlir/ir/FooOpInterfaces.h"
#include "foo/core/mlir/ir/FooPasses.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace foo {

namespace {
class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, FunctionPass> {
 public:
  void runOnFunction() override {
    auto f = getFunction();

    // Populate the worklist with the operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op)) opWorklist.insert(op);
    });

    // Iterate in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!opWorklist.empty()) {
      // Find the next operation ready for inference,
      // where all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end()) break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      // Ask the operation to infer its output shapes.
      DEBUG_WITH_TYPE("shape-inference",
                      llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError(
            "unable to infer shape of operation without shape "
            "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  static bool allOperandsInferred(Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
      return operandType.isa<RankedTensorType>();
    });
  }

  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return !resultType.isa<RankedTensorType>();
    });
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
}  // namespace foo
}  // namespace mlir
