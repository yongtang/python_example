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

import sys
import pytest

import foo


def test_simple():
    """test_simple"""

    def simple_fn():
        """simple_doc"""

    assert simple_fn.__doc__ == "simple_doc"
    assert simple_fn.__name__ == "simple_fn"


@pytest.mark.skipif(sys.platform == "win32", reason="TODO")
def test_print():
    """test_print"""

    @foo.jit
    def main():
        """main_doc"""
        print(1.0)

    check = """
    CHECK-LABEL: func @main() {
    CHECK-NEXT:   %0 = "foo.const"() {value = "1.0" : !foo.float} : () -> !foo.foo
    CHECK-NEXT:   foo.print %0 : !foo.foo
    CHECK-NEXT:   foo.return
    CHECK-NEXT: }
    """

    assert foo.check(main.mlir, check)

    main()
