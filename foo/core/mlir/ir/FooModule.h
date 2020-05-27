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

//===----------------------------------------------------------------------===//
//
// This file defines the module processing for Foo
//
//===----------------------------------------------------------------------===//

#ifndef FOO_CORE_MLIR_IR_FOOMODULE_H
#define FOO_CORE_MLIR_IR_FOOMODULE_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"

namespace mlir {
namespace foo {

int emitMLIR(mlir::MLIRContext &context, mlir::ModuleOp module,
             bool optimization, llvm::StringRef action);

int runJIT(mlir::ModuleOp module, bool optimization);

}  // namespace foo
}  // namespace mlir

#endif  // FOO_CORE_MLIR_IR_FOOMODULE_H
