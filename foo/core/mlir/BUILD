load("@org_tensorflow//third_party/mlir:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl(
    name = "foo_interface_inc_gen",
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "ir/FooOpInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "ir/FooOpInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/FooOpInterfaces.td",
    td_srcs = [
        "ir/FooOpInterfaces.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
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
        "ir/FooDialect.td",
        "ir/FooOpInterfaces.td",
        "ir/FooOps.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)

cc_library(
    name = "foo",
    srcs = [
        "ir/FooDialect.cpp",
        "ir/FooModule.cpp",
        "ir/FooOpInterfaces.cpp",
        "ir/FooOps.cpp",
        "ir/FooPassLowerToAffine.cpp",
        "ir/FooPassLowerToLLVM.cpp",
    ],
    hdrs = [
        "ir/FooDialect.h",
        "ir/FooDialect.h.inc",
        "ir/FooModule.h",
        "ir/FooOpInterfaces.cpp.inc",
        "ir/FooOpInterfaces.h",
        "ir/FooOpInterfaces.h.inc",
        "ir/FooOps.cpp.inc",
        "ir/FooOps.h",
        "ir/FooOps.h.inc",
        "ir/FooPasses.h",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    deps = [
        ":foo_inc_gen",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "foo_dialect_registration",
    srcs = [
        "ir/FooDialectRegistration.cpp",
    ],
    deps = [
        ":foo",
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

cc_binary(
    name = "fooc",
    srcs = [
        "tools/fooc.cpp",
    ],
    deps = [
        ":foo",
    ],
)

py_test(
    name = "ops_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/ops.mlir",
    ],
    data = [
        "tests/lit.cfg.py",
        "tests/ops.mlir",
        ":foo-opt",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "foo_mlir_tests",
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
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "foo_affine_mlir_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/foo.affine.mlir",
    ],
    data = [
        "tests/foo.affine.mlir",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "foo_llvm_mlir_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/foo.llvm.mlir",
    ],
    data = [
        "tests/foo.llvm.mlir",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "foo_llvm_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/foo.llvm",
    ],
    data = [
        "tests/foo.llvm",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "foo_jit_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/foo.jit",
    ],
    data = [
        "tests/foo.jit",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)
