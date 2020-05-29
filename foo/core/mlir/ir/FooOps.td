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
// This is the operation definition file for Foo dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef FOO_OPS
#define FOO_OPS

include "FooDialect.td"
include "FooOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Foo_FooOp : Foo_Op<"foo", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Illustrates how to define an operation.";
  let description = [{
    The `foo.foo` operation is a placeholder.
    This operation takes an integer argument and returns an integer.

    Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = foo.foo %0 : i32
        ```
  }];

  let arguments = (ins I32:$input);
  let results = (outs I32:$res);

  let assemblyFormat = [{$input attr-dict `:` type($input)}];
}
#endif  // FOO_OPS
