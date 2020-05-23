load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl(
    name = "foo_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/FooOps.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/FooOps.cpp.inc",
        ),
        (
            "-gen-dialect-decls",
            "ir/FooDialect.h.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/FooDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/FooOps.td",
    td_srcs = [
        "ir/FooOps.td",
        "ir/FooDialect.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)

cc_library(
    name = "foo",
    srcs = [
        "ir/FooDialect.cpp",
        "ir/FooDialect.h",
        "ir/FooDialect.h.inc",
        "ir/FooOps.cpp",
        "ir/FooOps.cpp.inc",
        "ir/FooOps.h",
        "ir/FooOps.h.inc",
    ],
    hdrs = [
    ],
    deps = [
        ":foo_inc_gen",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "foo-opt",
    srcs = [
        "tools/foo-opt.cpp",
    ],
    deps = [
        ":foo",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

py_test(
    name = "foo_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/foo.mlir",
    ],
    data = [
        "tests/foo.mlir",
        "tests/lit.cfg.py",
        ":foo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)
