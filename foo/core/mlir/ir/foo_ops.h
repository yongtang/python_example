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
// This file defines the dialect for Foo
//
//===----------------------------------------------------------------------===//

#ifndef FOO_CORE_MLIR_IR_FOO_OPS_H_
#define FOO_CORE_MLIR_IR_FOO_OPS_H_

#include "mlir/Dialect/Traits.h"          // from @llvm-project
#include "mlir/IR/Dialect.h"              // from @llvm-project
#include "mlir/IR/OpImplementation.h"     // from @llvm-project
#include "mlir/IR/StandardTypes.h"        // from @llvm-project
#include "mlir/Interfaces/SideEffects.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"            // from @llvm-project
namespace mlir {
namespace foo {

#include "foo/core/mlir/ir/foo_dialect.h.inc"

#define GET_OP_CLASSES
#include "foo/core/mlir/ir/foo_ops.h.inc"

}  // namespace foo
}  // namespace mlir

#endif  // FOO_CORE_MLIR_IR_FOO_OPS_H_
