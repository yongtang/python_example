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

#include "foo/core/mlir/ir/FooModule.h"
#include "foo/core/mlir/ir/FooOps.h"
#include "llvm/Support/FileCheck.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllDialects.h"

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

  py::class_<mlir::Location>(m, "Location");

  py::class_<mlir::Attribute>(m, "Attribute");
  py::class_<mlir::Type>(m, "Type");

  py::class_<mlir::Value>(m, "Value");

  py::class_<mlir::Block>(m, "Block").def("getArguments", [](mlir::Block& b) {
    auto arguments = b.getArguments();
    return std::vector<mlir::Value>(arguments.begin(), arguments.end());
  });

  py::class_<mlir::ModuleOp>(m, "ModuleOp")
      .def("create",
           [](mlir::Location loc) { return mlir::ModuleOp::create(loc); })
      .def("push_back",
           [](mlir::ModuleOp& m, mlir::FuncOp f) { m.push_back(f); })
      .def("emit",
           [](mlir::ModuleOp& m, mlir::MLIRContext& context) {
             if (int error =
                     mlir::foo::emitMLIR(m, true, "mlir-llvm", context)) {
               throw std::runtime_error("Unable to emit MLIR: " +
                                        std::to_string(error));
             }
           })
      .def("run",
           [](mlir::ModuleOp& m, std::string name, std::vector<double> args) {
             double args_in[1] = {args[0]};
             double args_out[1] = {0.0};
             void* data[2] = {&args_in[0], &args_out[0]};
             if (int error = mlir::foo::runJIT(
                     m, true, name, llvm::MutableArrayRef<void*>(data, 2))) {
               throw std::runtime_error("Unable to run JIT: " +
                                        std::to_string(error));
             }
             return args_out[0];
           })
      .def("__str__", [](mlir::ModuleOp& m) {
        std::string str;
        llvm::raw_string_ostream os(str);
        m.print(os);
        return os.str();
      });
  py::class_<mlir::Builder>(m, "Builder").def(py::init<mlir::MLIRContext*>());
  py::class_<mlir::OpBuilder, mlir::Builder>(m, "OpBuilder")
      .def(py::init<mlir::MLIRContext*>())
      .def(py::init<mlir::Region&>())
      .def(py::init<mlir::Operation*>())
      .def(py::init<mlir::Block*, mlir::Block::iterator>())
      .def("getContext", [](mlir::OpBuilder& b) { return b.getContext(); })
      .def("getUnknownLoc",
           [](mlir::OpBuilder& b) {
             return static_cast<mlir::Location>(b.getUnknownLoc());
           })
      .def("getFileLineColLoc",
           [](mlir::OpBuilder& b, std::string file, unsigned line,
              unsigned col) {
             return static_cast<mlir::Location>(
                 b.getFileLineColLoc(b.getIdentifier(file), line, col));
           })
      .def("getF64Type",
           [](mlir::OpBuilder& b) {
             return static_cast<mlir::Type>(b.getF64Type());
           })
      .def("getF64FloatAttr",
           [](mlir::OpBuilder& b, double value) {
             return static_cast<mlir::Attribute>(b.getF64FloatAttr(value));
           })
      .def("createReturnOp",
           [](mlir::OpBuilder& b, mlir::Location location,
              std::vector<mlir::Value> operands) {
             return b.create<mlir::ReturnOp>(location, operands);
           })
      .def("createConstantOp",
           [](mlir::OpBuilder& b, mlir::Location location,
              mlir::Attribute value) {
             return static_cast<mlir::Value>(
                 b.create<mlir::ConstantOp>(location, value));
           })
      .def("createAddFOp",
           [](mlir::OpBuilder& b, mlir::Location location, mlir::Value lhs,
              mlir::Value rhs) {
             return static_cast<mlir::Value>(
                 b.create<mlir::AddFOp>(location, lhs, rhs));
           })
      .def("setInsertionPointToStart",
           [](mlir::OpBuilder& b, mlir::Block* block) {
             return b.setInsertionPointToStart(block);
           });

  py::class_<mlir::FunctionType, mlir::Type>(m, "FunctionType")
      .def("get",
           [](std::vector<mlir::Type> inputs, std::vector<mlir::Type> results,
              mlir::MLIRContext* context) {
             return mlir::FunctionType::get(llvm::ArrayRef<mlir::Type>(inputs),
                                            llvm::ArrayRef<mlir::Type>(results),
                                            context);
           });

  py::class_<mlir::FuncOp>(m, "FuncOp")
      .def("create",
           [](mlir::Location location, std::string name,
              mlir::FunctionType& type) {
             return mlir::FuncOp::create(location, name, type);
           })
      .def("setType", [](mlir::FuncOp& f,
                         mlir::FunctionType type) { return f.setType(type); })
      .def(
          "addEntryBlock", [](mlir::FuncOp& f) { return f.addEntryBlock(); },
          py::return_value_policy::reference);
  py::class_<mlir::ReturnOp>(m, "ReturnOp")
      .def("getOperandTypes", [](mlir::ReturnOp& o) {
        auto operandTypes = o.getOperandTypes();
        std::vector<mlir::Type> types(operandTypes.begin(), operandTypes.end());
        return types;
      });
}
