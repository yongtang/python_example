# Copyright 2020 Yong Tang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""foo"""

import os
import sys
import ast
import inspect
import textwrap

if "FOO_BINDIR" in os.environ:
    path = os.path.abspath(
        os.path.join(os.environ["FOO_BINDIR"], "foo", "core", "python", "pybind")
    )
else:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pybind")
sys.path.insert(0, path)
import pybind_mlir


class MLIRNodeVisitor(ast.NodeVisitor):
    def __init__(self, builder):
        self.builder = builder

    def visit_FunctionDef(self, node: ast.FunctionDef):
        location = self.builder.getFileLineColLoc("mlir", node.lineno, node.col_offset)
        inputs, outputs = [], []
        function = pybind_mlir.FuncOp.create(
            location, node.name, self.builder.getFunctionType(inputs, outputs)
        )

        entry_block = function.addEntryBlock()
        self.builder.setInsertionPointToStart(entry_block)

        items = [self.visit(e) for e in node.body]
        if len(items) == 0 or not isinstance(items[-1], pybind_mlir.ReturnOp):
            self.builder.createReturnOp(location, [])
        else:
            return_op = items[-1]
            function.setType(
                self.builder.getFunctionType([], return_op.getOperandTypes())
            )

        return function

    def visit_Return(self, node: ast.Return):
        location = self.builder.getFileLineColLoc("mlir", node.lineno, node.col_offset)
        value = None if node.value is None else self.visit(node.value)

        operands = [] if value is None else value.getOperation().getResults()
        return self.builder.createReturnOp(location, operands)

    def visit_Num(self, node: ast.Num):
        location = self.builder.getFileLineColLoc("mlir", node.lineno, node.col_offset)
        type = self.builder.getOpaqueType("foo", "opaque")
        attribute = self.builder.getOpaqueAttr("foo", str(node.n), type)

        return self.builder.createConstantOp(location, type, attribute)


class Function:
    """Function"""

    def __init__(self, function, signature=None):
        self._function = function
        self._signature = signature

        code = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(code)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)

        node = tree.body[0]

        context = pybind_mlir.MLIRContext()
        builder = pybind_mlir.OpBuilder(context)

        func = MLIRNodeVisitor(builder).visit(node)

        module = pybind_mlir.ModuleOp.create(builder.getUnknownLoc(),)
        module.push_back(func)

        self._context = context
        self._module = module

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    @property
    def __doc__(self):
        return self._function.__doc__

    @property
    def __name__(self):
        return self._function.__name__

    @property
    def signature(self):
        return self._signature

    def check(self, check):
        return pybind_mlir.check(str(self._module), check)


def jit(signature_or_function=None):
    def _jit(function):
        return Function(function)

    return Function(signature_or_function) if callable(signature_or_function) else _jit
