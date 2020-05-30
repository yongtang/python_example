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
    def simple_fn(i):
        """simple_doc"""
        return i + 1.0

    check = """
    CHECK-LABEL: func @simple_fn(%arg0: f64) -> f64 {
    CHECK-NEXT:   %cst = constant 1.000000e+00 : f64
    CHECK-NEXT:   %0 = addf %arg0, %cst : f64
    CHECK-NEXT:   return %0 : f64
    CHECK-NEXT: }
    """

    assert simple_fn.__doc__ == "simple_doc"
    assert simple_fn.__name__ == "simple_fn"

    assert foo.check(simple_fn.mlir, check)

    assert simple_fn(1.0) == 2.0
