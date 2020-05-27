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

#include "foo/core/mlir/ir/FooOps.h"

#include "foo/core/mlir/ir/FooDialect.h"
#include "foo/core/mlir/ir/FooOpInterfaces.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/OpImplementation.h"
namespace mlir {
namespace foo {

namespace {
static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}
static mlir::LogicalResult verifyReturnOp(ReturnOp op) {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>(op.getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (op.getNumOperands() > 1)
    return op.emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getType().getResults();
  if (op.getNumOperands() != results.size())
    return op.emitOpError()
           << "does not return the same number of values ("
           << op.getNumOperands() << ") as the enclosing function ("
           << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!op.hasOperand()) return mlir::success();

  auto inputType = *op.operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return op.emitError() << "type of return operand ("
                        << *op.operand_type_begin()
                        << ") doesn't match function result type ("
                        << results.front() << ")";
}

}  // namespace

void ConstOp::inferTypes() {
  if (getResult().getType().isa<mlir::OpaqueType>()) {
    if (auto opaque = getValue().getType().dyn_cast<mlir::OpaqueType>()) {
      if (opaque.getTypeData() == "float") {
        getResult().setType(FloatType::getF64(getContext()));
      }
    }
  }
}

void ConstOp::inferShapes() {
  if (!getResult().getType().dyn_cast<mlir::RankedTensorType>()) {
    if (auto opaque = getValue().getType().dyn_cast<mlir::OpaqueType>()) {
      if (getResult().getType().isa<mlir::FloatType>()) {
        double number;

        if (to_float(getValue().dyn_cast<mlir::StringAttr>().getValue(),
                     number)) {
          auto type = mlir::RankedTensorType::get({}, getResult().getType());
          auto value = mlir::DenseElementsAttr::get(type, {number});
          getResult().setType(type);

          setValue(value);
        }
      }
    }
  }
}

#define GET_OP_CLASSES
#include "foo/core/mlir/ir/FooOps.cpp.inc"
}  // namespace foo
}  // namespace mlir
