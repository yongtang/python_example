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
            "-gen-op-interface-decls",
            "ir/FooOpInterfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "ir/FooOpInterfaces.cpp.inc",
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
        "ir/FooOpInterfaces.cpp",
        "ir/FooOps.cpp",
        "ir/FooPassLowerToAffineLoops.cpp",
        "ir/FooPassLowerToLLVM.cpp",
        "ir/FooPassShapeInference.cpp",
    ],
    hdrs = [
        "ir/FooDialect.h",
        "ir/FooDialect.h.inc",
        "ir/FooOpInterfaces.cpp.inc",
        "ir/FooOpInterfaces.h",
        "ir/FooOpInterfaces.h.inc",
        "ir/FooOps.cpp.inc",
        "ir/FooOps.h",
        "ir/FooOps.h.inc",
        "ir/FooPasses.h",
    ],
    deps = [
        ":foo_inc_gen",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:AffineToStandardTransforms",
        "@llvm-project//mlir:CFGTransforms",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMTransforms",
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

cc_binary(
    name = "fooc",
    srcs = [
        "tools/fooc.cpp",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:darwin": [],
        "@bazel_tools//src/conditions:windows": [],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    deps = [
        ":foo",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
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
    name = "fooc_mlir_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/fooc_mlir.mlir",
    ],
    data = [
        "tests/fooc_mlir.mlir",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "fooc_mlir_affine_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/fooc_mlir_affine.mlir",
    ],
    data = [
        "tests/fooc_mlir_affine.mlir",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "fooc_mlir_llvm_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/fooc_mlir_llvm.mlir",
    ],
    data = [
        "tests/fooc_mlir_llvm.mlir",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "fooc_llvm_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/fooc_llvm.mlir",
    ],
    data = [
        "tests/fooc_llvm.mlir",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)

py_test(
    name = "fooc_jit_tests",
    srcs = [
        "@llvm-project//llvm:lit",
    ],
    args = [
        "-v",
        "foo/core/mlir/tests/fooc_jit.mlir",
    ],
    data = [
        "tests/fooc_jit.mlir",
        "tests/lit.cfg.py",
        ":fooc",
        "@llvm-project//llvm:FileCheck",
    ],
    main = "lit.py",
)
