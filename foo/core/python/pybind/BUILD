licenses(["notice"])  # Apache 2.0

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "pybind_mlir",
    srcs = [
        "mlir.cc",
    ],
    copts = [],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)
