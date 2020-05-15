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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llvm/Support/FileCheck.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"

namespace py = pybind11;

PYBIND11_MODULE(pybind_mlir, m) {
  m.def("check", [](std::string input, std::string check) {
    llvm::FileCheckRequest fcr;
    llvm::FileCheck fc(fcr);
    llvm::SourceMgr SM = llvm::SourceMgr();
    SM.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(input),
                          llvm::SMLoc());
    SM.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(check),
                          llvm::SMLoc());
    llvm::Regex regex = fc.buildCheckPrefixRegex();
    fc.readCheckFile(SM, llvm::StringRef(check), regex);
    return fc.checkInput(SM, llvm::StringRef(input));
  });

  py::class_<mlir::MLIRContext>(m, "MLIRContext").def(py::init<>());

  py::class_<mlir::Value>(m, "Value");

  py::class_<mlir::Location>(m, "Location");
  py::class_<mlir::UnknownLoc>(m, "UnknownLoc");
  py::class_<mlir::FileLineColLoc>(m, "FileLineColLoc");

  py::class_<mlir::Builder>(m, "Builder").def(py::init<mlir::MLIRContext*>());

  py::class_<mlir::Type>(m, "Type");
  py::class_<mlir::FunctionType, mlir::Type>(m, "FunctionType");
  py::class_<mlir::OpaqueType, mlir::Type>(m, "OpaqueType");

  py::class_<mlir::Attribute>(m, "Attribute");
  py::class_<mlir::OpaqueAttr, mlir::Attribute>(m, "OpaqueAttr");

  py::class_<mlir::Block>(m, "Block");

  py::class_<mlir::Operation, std::unique_ptr<mlir::Operation, py::nodelete>>(
      m, "Operation")
      .def("getName",
           [](mlir::Operation& o) { return o.getName().getStringRef().str(); })
      .def("getDialect",
           [](mlir::Operation& o) { return o.getName().getDialect().str(); })
      .def("getResults", [](mlir::Operation& o) {
        auto results = o.getResults();
        return std::vector<mlir::Value>(results.begin(), results.end());
      });

  py::class_<mlir::OpBuilder, mlir::Builder>(m, "OpBuilder")
      .def(py::init<mlir::MLIRContext*>())
      .def(py::init<mlir::Region&>())
      .def(py::init<mlir::Operation*>())
      .def(py::init<mlir::Block*, mlir::Block::iterator>())
      .def("getUnknownLoc", &mlir::OpBuilder::getUnknownLoc)
      .def("getFileLineColLoc",
           [](mlir::OpBuilder& b, std::string file, unsigned line,
              unsigned col) {
             return b.getFileLineColLoc(b.getIdentifier(file), line, col);
           })
      .def("getFunctionType",
           [](mlir::OpBuilder& b, std::vector<mlir::Type> inputs,
              std::vector<mlir::Type> outputs) {
             return b.getFunctionType(llvm::ArrayRef<mlir::Type>(inputs),
                                      llvm::ArrayRef<mlir::Type>(outputs));
           })
      .def("getOpaqueType",
           [](mlir::OpBuilder& b, std::string dialect, std::string data) {
             return mlir::OpaqueType::get(
                 mlir::Identifier::get(dialect, b.getContext()), data,
                 b.getContext());
           })
      .def("getOpaqueAttr",
           [](mlir::OpBuilder& b, std::string dialect, std::string data,
              mlir::Type type) {
             return mlir::OpaqueAttr::get(
                 mlir::Identifier::get(dialect, b.getContext()), data, type,
                 b.getContext());
           })
      .def("setInsertionPointToStart",
           [](mlir::OpBuilder& b, mlir::Block* block) {
             return b.setInsertionPointToStart(block);
           })
      .def("createConstantOp",
           [](mlir::OpBuilder& b, mlir::Location location, mlir::Type type,
              mlir::Attribute attribute) {
             return b.create<mlir::ConstantOp>(location, type, attribute);
           })
      .def("createReturnOp", [](mlir::OpBuilder& b, mlir::Location location,
                                std::vector<mlir::Value> operands) {
        return b.create<mlir::ReturnOp>(location, operands);
      });

  py::class_<mlir::ModuleOp>(m, "ModuleOp")
      .def("create",
           [](mlir::Location loc) { return mlir::ModuleOp::create(loc); })
      .def("push_back",
           [](mlir::ModuleOp& m, mlir::FuncOp f) { m.push_back(f); })
      .def("dump", &mlir::ModuleOp::dump)
      .def("__str__", [](mlir::ModuleOp& m) {
        std::string str;
        llvm::raw_string_ostream os(str);
        m.print(os);
        return os.str();
      });

  py::class_<mlir::FuncOp>(m, "FuncOp")
      .def("create",
           [](mlir::Location location, std::string name,
              mlir::FunctionType type) {
             return mlir::FuncOp::create(location, name, type);
           })
      .def("setType", [](mlir::FuncOp& f,
                         mlir::FunctionType type) { return f.setType(type); })
      .def(
          "addEntryBlock", [](mlir::FuncOp& f) { return f.addEntryBlock(); },
          py::return_value_policy::reference);

  py::class_<mlir::ConstantOp>(m, "ConstantOp")
      .def("getOperation", &mlir::ConstantOp::getOperation);

  py::class_<mlir::ReturnOp>(m, "ReturnOp")
      .def("getOperation", &mlir::ReturnOp::getOperation)
      .def("getOperandTypes", [](mlir::ReturnOp& o) {
        auto operandTypes = o.getOperandTypes();
        std::vector<mlir::Type> types(operandTypes.begin(), operandTypes.end());

        return types;
      });
}
