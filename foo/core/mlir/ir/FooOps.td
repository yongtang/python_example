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

def Foo_ConstantOp : Foo_Op<"constant", [NoSideEffect]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value.
    The data is attached to the operation as an attribute.

    Example:

    ```mlir
      %0 = foo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Specify a parser and printer method.
  let parser = [{ return parseConstantOp(parser, result); }];
  let printer = [{ return printConstantOp(p, *this); }];

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<"OpBuilder &builder, OperationState &state, "
              "DenseElementsAttr value",
              [{ build(builder, state, value.getType(), value); }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"OpBuilder &builder, OperationState &state, double value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return verifyConstantOp(*this); }];
}

def Foo_PrintOp : Foo_Op<"print"> {
  let summary = "print operation";
  let description = [{
      The "print" builtin operation prints a given input tensor,
      and produces no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Foo_ReturnOp
    : Foo_Op<"return", [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation.

    Example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        foo.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [OpBuilder<"OpBuilder &b, OperationState &state",
                            [{ build(b, state, llvm::None); }]>];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return verifyReturnOp(*this); }];
}
#endif  // FOO_OPS
