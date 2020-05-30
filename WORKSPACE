workspace(name = "foo")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "zlib",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
)

http_archive(
    name = "pybind11_bazel",
    patch_cmds = [
        """sed -i.bak 's/"python"/"python3"/g' python_configure.bzl""",
    ],
    patch_cmds_win = [
        """echo "No patch on Windows" """,
    ],
    sha256 = "883042b7560af64bde10822c2c7f9de5c662e5f6ae2072045d8b83ff66b6cf86",
    strip_prefix = "pybind11_bazel-16ed1b8f308d2b3dec9d7e6decaad49ce4d28b43",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11_bazel/archive/16ed1b8f308d2b3dec9d7e6decaad49ce4d28b43.tar.gz",
        "https://github.com/pybind/pybind11_bazel/archive/16ed1b8f308d2b3dec9d7e6decaad49ce4d28b43.tar.gz",
    ],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "97504db65640570f32d3fdf701c25a340c8643037c3b69aec469c10c93dc8504",
    strip_prefix = "pybind11-2.5.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.5.0.tar.gz",
        "https://github.com/pybind/pybind11/archive/v2.5.0.tar.gz",
    ],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "7b9bbe3ea1fccb46dcfa6c3f3e29ba7ec740d8733370e21cdc8937467b4a4349",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.22.4/rules_go-v0.22.4.tar.gz",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.22.4/rules_go-v0.22.4.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains()

http_archive(
    name = "bazel_gazelle",
    sha256 = "d8c45ee70ec39a57e7a05e5027c32b1576cc7f16d9dd37135b0eddde45cf1b10",
    urls = [
        "https://storage.googleapis.com/bazel-mirror/github.com/bazelbuild/bazel-gazelle/releases/download/v0.20.0/bazel-gazelle-v0.20.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.20.0/bazel-gazelle-v0.20.0.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

http_archive(
    name = "com_google_protobuf",
    sha256 = "cfcba2df10feec52a84208693937c17a4b5df7775e1635c1e3baffc487b24c9b",
    strip_prefix = "protobuf-3.9.2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
        "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

http_archive(
    name = "com_github_bazelbuild_buildtools",
    sha256 = "d5558cd419c8d46bdc958064cb97f963d1ea793866414c025906ec15033512ed",
    strip_prefix = "buildtools-3.0.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/buildtools/archive/v3.0.0.tar.gz",
        "https://github.com/bazelbuild/buildtools/archive/v3.0.0.tar.gz",
    ],
)

http_archive(
    name = "rules_python",
    sha256 = "b5668cde8bb6e3515057ef465a35ad712214962f0b3a314e551204266c7be90c",
    strip_prefix = "rules_python-0.0.2",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.2/rules_python-0.0.2.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip3_import")

pip3_import(
    name = "lint_dependencies",
    requirements = "//tools/lint:requirements.txt",
)

load("@lint_dependencies//:requirements.bzl", "pip_install")

pip_install()

http_archive(
    name = "com_grail_bazel_toolchain",
    sha256 = "b060a4f28c7a03485a914e9c03683ddaf6792edb5d72c91f82a28f393b1bbb0b",
    strip_prefix = "bazel-toolchain-f4c17a3ae40f927ff62cc0fb8fe22b1530871807",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grailbio/bazel-toolchain/archive/f4c17a3ae40f927ff62cc0fb8fe22b1530871807.tar.gz",
        "https://github.com/grailbio/bazel-toolchain/archive/f4c17a3ae40f927ff62cc0fb8fe22b1530871807.tar.gz",
    ],
)

load("@com_grail_bazel_toolchain//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "10.0.0",
)

http_archive(
    name = "org_tensorflow",
    patch_cmds = [
        """rm WORKSPACE BUILD tensorflow/BUILD""",
        """echo '' > BUILD""",
        """echo 'config_setting(name = "macos", values = {"apple_platform_type": "macos", "cpu": "darwin"}, visibility = ["//visibility:public"])' >> tensorflow/BUILD""",
        """echo 'config_setting(name = "windows", values = {"cpu": "x64_windows"}, visibility = ["//visibility:public"])' >> tensorflow/BUILD""",
        """echo 'config_setting(name = "freebsd", values = {"cpu": "freebsd"}, visibility = ["//visibility:public"])' >> tensorflow/BUILD""",
        """echo 'config_setting(name = "linux_ppc64le", values = {"cpu": "ppc"}, visibility = ["//visibility:public"])' >> tensorflow/BUILD""",
        #"""sed -i.bak 's/cmd = ("/cmd = ("python3 /g' third_party/llvm/llvm.bzl""",
    ],
    sha256 = "6d7d649c6a0bd2e4bcb805d62ec4060d23e7e42901c4d535e649cd1cc1490980",
    strip_prefix = "tensorflow-e2dfc382e6be58fff6ee6d0969f8925e531ac998",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/tensorflow/tensorflow/archive/e2dfc382e6be58fff6ee6d0969f8925e531ac998.tar.gz",
        "https://github.com/tensorflow/tensorflow/archive/e2dfc382e6be58fff6ee6d0969f8925e531ac998.tar.gz",
    ],
)

load("@org_tensorflow//third_party:repo.bzl", "tf_http_archive")

tf_http_archive(
    name = "llvm-project",
    additional_build_files = {
        "@org_tensorflow//third_party/llvm:llvm.autogenerated.BUILD": "llvm/BUILD",
        "@org_tensorflow//third_party/mlir:BUILD": "mlir/BUILD",
        "@org_tensorflow//third_party/mlir:test.BUILD": "mlir/test/BUILD",
    },
    sha256 = "d7e67036dc89906cb2f80df7b0b7de6344d86eddf6e98bb4d01a578242889a73",
    strip_prefix = "llvm-project-b726d071b4aa46004228fc38ee5bfd167f999bfe",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/b726d071b4aa46004228fc38ee5bfd167f999bfe.tar.gz",
        "https://github.com/llvm/llvm-project/archive/b726d071b4aa46004228fc38ee5bfd167f999bfe.tar.gz",
    ],
)
