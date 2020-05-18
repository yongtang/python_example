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

#ifndef FOO_DIALECT
#define FOO_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Foo dialect definitions
//===----------------------------------------------------------------------===//

def FooDialect : Dialect {
  let name = "foo";

  let summary = "Types and operations for Foo dialect";
  let description = [{
    This dialect contains operations for Foo. This dialect will be
    used in conjunction with other dialects for converting & optimizing.
  }];

  let cppNamespace = "foo";
}

//===----------------------------------------------------------------------===//
// Foo op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class Foo_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<FooDialect, mnemonic, traits>;

def Foo_ConstOp : Foo_Op<"const", [ConstantLike, NoSideEffect]> {
  let summary = "Const operator";
  let description = [{
      Const preparation operator
  }];

  let arguments = (ins AnyAttr : $value);
  let results = (outs AnyType);

  let builders =
      [OpBuilder<"OpBuilder &builder, OperationState &result, Attribute value",
                 [{ build(builder, result, value.getType(), value); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }

    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  // let hasFolder = 1;
}

def Foo_UnaryOp : Foo_Op<"unary", [NoSideEffect]> {
  let summary = "Unary operator";
  let description = [{
      Unary preparation operator
  }];

  let arguments = (ins AnyAttr : $kind, AnyType : $operand);
  let results = (outs AnyType);

  // let hasFolder = 1;
}

def Foo_BinaryOp : Foo_Op<"binary", [NoSideEffect]> {
  let summary = "Binary operator";
  let description = [{
      Binary preparation operator
  }];

  let arguments = (ins AnyAttr : $kind, AnyType : $left, AnyType : $right);
  let results = (outs AnyType);

  // let hasFolder = 1;
}

#endif  // FOO_DIALECT
