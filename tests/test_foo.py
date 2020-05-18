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
"""test_foo"""

import foo


def test_simple():
    """test_simple"""

    @foo.jit
    def simple_fn():
        """simple_doc"""

    mlir_exp = r"""
      CHECK-LABEL: func @simple_fn() {
      CHECK-NEXT:   "std.return"() : () -> ()
      CHECK-NEXT: }
    """
    assert simple_fn.__doc__ == "simple_doc"
    assert simple_fn.__name__ == "simple_fn"
    assert simple_fn.signature is None
    assert simple_fn.check(mlir_exp)


def test_function():
    """test_function"""

    @foo.jit
    def fn_const():
        return 5

    exp_const = r"""
      CHECK-LABEL: func @fn_const() -> !foo.int {
      CHECK-NEXT:   %0 = "foo.const"() {value = "5" : !foo.int} : () -> !foo.int
      CHECK-NEXT:   "std.return"(%0) : (!foo.int) -> ()
      CHECK-NEXT: }
    """
    assert fn_const.check(exp_const)

    @foo.jit
    def fn_unary():
        return -5

    exp_unary = r"""
      CHECK-LABEL: func @fn_unary() -> !foo.opaque {
      CHECK-NEXT:   %0 = "foo.const"() {value = "5" : !foo.int} : () -> !foo.int
      CHECK-NEXT:   %1 = "foo.unary"(%0) {kind = "usub" : !foo.op} : (!foo.int) -> !foo.opaque
      CHECK-NEXT:   "std.return"(%1) : (!foo.opaque) -> ()
      CHECK-NEXT: }
    """
    assert fn_unary.check(exp_unary)

    @foo.jit
    def fn_binary():
        return 3 + 5

    exp_binary = r"""
      CHECK-LABEL: func @fn_binary() -> !foo.opaque {
      CHECK-NEXT:   %0 = "foo.const"() {value = "3" : !foo.int} : () -> !foo.int
      CHECK-NEXT:   %1 = "foo.const"() {value = "5" : !foo.int} : () -> !foo.int
      CHECK-NEXT:   %2 = "foo.binary"(%0, %1) {kind = "add" : !foo.op} : (!foo.int, !foo.int) -> !foo.opaque
      CHECK-NEXT:   "std.return"(%2) : (!foo.opaque) -> ()
      CHECK-NEXT: }
    """
    assert fn_binary.check(exp_binary)
