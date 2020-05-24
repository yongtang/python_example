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
// This file contains the declarations of the FooOpInterfaces.td
//
//===----------------------------------------------------------------------===//

#ifndef FOO_CORE_MLIR_IR_FOOOPINTERFACES_H_
#define FOO_CORE_MLIR_IR_FOOOPINTERFACES_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace foo {

/// Include the auto-generated declarations.
#include "foo/core/mlir/ir/FooOpInterfaces.h.inc"

}  // end namespace foo
}  // end namespace mlir

#endif  // FOO_CORE_MLIR_IR_FOOOPINTERFACES_H_
