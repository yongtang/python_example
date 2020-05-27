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
#include "foo/core/mlir/ir/FooModule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"

namespace {

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module,
             llvm::StringRef inputFilename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  return 0;
}

int dumpLLVM(mlir::OwningModuleRef &module, bool optimization) {
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
      /*optLevel=*/optimization ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::outs() << *llvmModule << "\n";
  return 0;
}
}  // namespace

int main(int argc, char **argv) {
  enum EmitAction {
    None = -1,
    EmitMLIR,
    EmitMLIRInference,
    EmitMLIRAffine,
    EmitMLIRLLVM,
    EmitLLVMIR,
    EmitRunJIT
  };
  llvm::SmallVector<llvm::StringRef, 6> emitMLIRChoice({
      "mlir",
      "mlir-inference",
      "mlir-affine",
      "mlir-llvm",
      "mlir-llvm",
      "mlir-llvm",
  });

  llvm::cl::opt<enum EmitAction> emitAction(
      "emit", llvm::cl::desc("Select the kind of output desired"),
      llvm::cl::values((clEnumValN(EmitMLIR, "mlir", "output the MLIR dump"))),
      llvm::cl::values(clEnumValN(EmitMLIRInference, "mlir-inference",
                                  "output the MLIR dump after inference")),
      llvm::cl::values(
          clEnumValN(EmitMLIRAffine, "mlir-affine",
                     "output the MLIR dump after affine lowering")),
      llvm::cl::values(clEnumValN(EmitMLIRLLVM, "mlir-llvm",
                                  "output the MLIR dump after llvm lowering")),
      llvm::cl::values(
          clEnumValN(EmitLLVMIR, "llvm", "output the LLVM IR dump")),
      llvm::cl::values(
          clEnumValN(EmitRunJIT, "jit",
                     "JIT the code and run it by invoking the main function")));

  llvm::cl::opt<bool> enableOptimization(
      "opt", llvm::cl::desc("Enable optimizations"));

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input foo file>"),
      llvm::cl::init("-"), llvm::cl::value_desc("filename"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "foo compiler\n");

  mlir::registerAllDialects();
  mlir::registerDialect<mlir::foo::FooDialect>();

  mlir::registerPassManagerCLOptions();

  mlir::MLIRContext context;
  mlir::OwningModuleRef module;

  if (int error = loadMLIR(context, module, inputFilename)) {
    return error;
  }

  if (int error = mlir::foo::emitMLIR(context, *module, enableOptimization,
                                      emitMLIRChoice[emitAction])) {
    return error;
  }

  if (emitAction == EmitAction::EmitLLVMIR) {
    return dumpLLVM(module, enableOptimization);
  }

  if (emitAction == EmitAction::EmitRunJIT) {
    return mlir::foo::runJIT(*module, enableOptimization);
  }

  std::string str;
  llvm::raw_string_ostream os(str);
  module->print(os);
  llvm::outs() << os.str() << "\n";
  return 0;
}
