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
"""Setup for pip package."""

import os
import sys
import shutil
import tempfile
import fnmatch
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

project = "foo"
version = "0.1.0"

datapath = os.environ.get("FOO_BINDIR", None)

if (datapath is not None) and ("bdist_wheel" in sys.argv):
    rootpath = tempfile.mkdtemp()
    print("setup.py - create {} and copy data files".format(rootpath))
    for rootname, _, filenames in os.walk(os.path.join(datapath, "foo")):
        if not fnmatch.fnmatch(rootname, "*test*") and not fnmatch.fnmatch(
            rootname, "*runfiles*"
        ):
            for filename in [
                f
                for f in filenames
                if fnmatch.fnmatch(f, "*.pyd")
                or fnmatch.fnmatch(f, "*.so")
                or fnmatch.fnmatch(f, "*.py")
            ]:
                src = os.path.join(rootname, filename)
                dst = os.path.join(
                    rootpath,
                    os.path.relpath(os.path.join(rootname, filename), datapath),
                )
                print("setup.py - copy {} to {}".format(src, dst))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
    sys.argv.append("--bdist-dir")
    sys.argv.append(rootpath)


class BinaryDistribution(setuptools.dist.Distribution):
    def has_ext_modules(self):
        return True


setuptools.setup(
    name=project,
    version=version,
    packages=setuptools.find_packages(where=".", exclude=["tests"]),
    python_requires=">=3.5, <3.9",
    zip_safe=False,
    distclass=BinaryDistribution,
)
