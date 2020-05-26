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
// This file defines the dialect ops for Foo
//
//===----------------------------------------------------------------------===//

#ifndef FOO_CORE_MLIR_IR_FOOOPS_H
#define FOO_CORE_MLIR_IR_FOOOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "foo/core/mlir/ir/FooOpInterfaces.h"
#include "foo/core/mlir/ir/FooDialect.h"

namespace mlir {
namespace foo {

#define GET_OP_CLASSES
#include "foo/core/mlir/ir/FooOps.h.inc"

}  // namespace foo
}  // namespace mlir

#endif  // FOO_CORE_MLIR_IR_FOOOPS_H
