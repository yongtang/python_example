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

#include "foo/core/mlir/ir/FooModule.h"

#include "foo/core/mlir/ir/FooPasses.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace foo {

int emitMLIR(mlir::ModuleOp module, bool optimization, llvm::StringRef action,
             mlir::MLIRContext &context) {
  enum EmitAction {
    EmitMLIR,
    EmitMLIRAffine,
    EmitMLIRLLVM,
  } emitAction;

  if (action.str() == "mlir") {
    emitAction = EmitMLIR;
  } else if (action.str() == "mlir-affine") {
    emitAction = EmitMLIRAffine;
  } else if (action.str() == "mlir-llvm") {
    emitAction = EmitMLIRLLVM;
  } else {
    llvm::errs() << "Error can't run mlir module with: " << action.str()
                 << "\n";
    return 5;
  }

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  if (optimization || emitAction > EmitAction::EmitMLIR) {
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    // Only one function now, infer type/shape.
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (emitAction >= EmitAction::EmitMLIRAffine) {
    // Lower to affine dialect with clean up afterwards.
    pm.addPass(mlir::foo::createLowerToAffinePass());

    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createSCCPPass());

    // Add optimizations if enabled.
    if (optimization) {
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createMemRefDataFlowOptPass());
    }
  }

  if (emitAction >= EmitAction::EmitMLIRLLVM) {
    // Lower to LLVM dialect.
    pm.addPass(mlir::foo::createLowerToLLVMPass());
  }

  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Error can't run mlir module\n";
    return 4;
  }

  return 0;
}

int runJIT(mlir::ModuleOp module, bool optimization, llvm::StringRef name,
           llvm::MutableArrayRef<void *> args) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/optimization ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(module, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invoke(name, args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

}  // namespace foo
}  // namespace mlir
