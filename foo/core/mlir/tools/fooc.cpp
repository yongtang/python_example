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
#include "foo/core/mlir/ir/FooPasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

namespace {
enum EmitAction {
  None,
  EmitMLIR,
  EmitMLIRAffine,
  EmitMLIRLLVM,
  EmitLLVMIR,
  EmitRunJIT
};

static llvm::cl::opt<enum EmitAction> emitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(EmitMLIR, "mlir", "output the MLIR dump")),
    llvm::cl::values(clEnumValN(EmitMLIRAffine, "mlir-affine",
                                "output the MLIR dump after affine lowering")),
    llvm::cl::values(clEnumValN(EmitMLIRLLVM, "mlir-llvm",
                                "output the MLIR dump after llvm lowering")),
    llvm::cl::values(clEnumValN(EmitLLVMIR, "llvm", "output the LLVM IR dump")),
    llvm::cl::values(
        clEnumValN(EmitRunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

static llvm::cl::opt<bool> enableOptimization(
    "opt", llvm::cl::desc("Enable optimizations"));

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input foo file>"),
    llvm::cl::init("-"), llvm::cl::value_desc("filename"));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  if (enableOptimization || emitAction >= EmitAction::EmitMLIRAffine) {
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::foo::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (emitAction >= EmitAction::EmitMLIRAffine) {
    // Partially lower the foo dialect with a few cleanups afterwards.
    pm.addPass(mlir::foo::createLowerToAffinePass());

    mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(mlir::createSCCPPass());

    // Add optimizations if enabled.
    if (enableOptimization) {
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createMemRefDataFlowOptPass());
    }
  }

  if (emitAction >= EmitAction::EmitMLIRLLVM) {
    // Finish lowering the foo IR to the LLVM dialect.
    pm.addPass(mlir::foo::createLowerToLLVMPass());
  }

  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Error can't run mlir module\n";
    return 4;
  }

  return 0;
}

int dumpLLVM(mlir::OwningModuleRef &module) {
  auto llvmModule = mlir::translateModuleToLLVMIR(*module);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOptimization ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::outs() << *llvmModule << "\n";
  return 0;
}

int runJIT(mlir::OwningModuleRef &module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOptimization ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(*module, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invoke("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "foo compiler\n");

  mlir::registerAllDialects();
  mlir::registerPassManagerCLOptions();

  mlir::registerDialect<mlir::foo::FooDialect>();

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;

  if (int error = loadMLIR(context, module)) {
    return error;
  }

  if (emitAction == EmitAction::EmitLLVMIR) {
    return dumpLLVM(module);
  }

  if (emitAction == EmitAction::EmitRunJIT) {
    return runJIT(module);
  }

  std::string str;
  llvm::raw_string_ostream os(str);
  module->print(os);
  llvm::outs() << os.str() << "\n";
  return 0;
}
