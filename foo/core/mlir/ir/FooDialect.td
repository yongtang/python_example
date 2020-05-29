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
// This is the operation definition file for Foo dialect base operations.
//
//===----------------------------------------------------------------------===//

#ifndef FOO_DIALECT
#define FOO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Foo dialect definition.
//===----------------------------------------------------------------------===//

def Foo_Dialect : Dialect {
  let name = "foo";
  let summary = "A foo MLIR dialect.";
  let description = [{This dialect is a placeholder MLIR dialect.}];
  let cppNamespace = "foo";
}

//===----------------------------------------------------------------------===//
// Base foo operation definition.
//===----------------------------------------------------------------------===//

class Foo_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Foo_Dialect, mnemonic, traits>;

#endif  // FOO_DIALECT
