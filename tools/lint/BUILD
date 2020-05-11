load("def.bzl", "black", "clang", "pyupgrade")
load("@lint_dependencies//:requirements.bzl", "requirement")
load("@com_github_bazelbuild_buildtools//buildifier:def.bzl", "buildifier")

clang(
    name = "clang",
)

black(
    name = "black",
)

pyupgrade(
    name = "pyupgrade",
)

buildifier(
    name = "buildifier",
)

genrule(
    name = "clang_binary",
    srcs = ["@llvm_toolchain//:bin/clang-format"],
    outs = ["clang-binary"],
    cmd = "cat $< > $@",
    executable = True,
)

genrule(
    name = "black_binary",
    srcs = [],
    outs = ["black-binary"],
    cmd = "echo '$(location :black_py) \"$$@\"' > $@",
    executable = True,
    tools = [
        ":black_py",
    ],
)

genrule(
    name = "pyupgrade_binary",
    srcs = [],
    outs = ["pyupgrade-binary"],
    cmd = "echo '$(location :pyupgrade_py) \"$$@\"' > $@",
    executable = True,
    tools = [
        ":pyupgrade_py",
    ],
)

py_binary(
    name = "black_py",
    srcs = ["black_python.py"],
    main = "black_python.py",
    deps = [
        requirement("black"),
        requirement("attrs"),
        requirement("regex"),
        requirement("appdirs"),
        requirement("click"),
        requirement("toml"),
        requirement("typed_ast"),
        requirement("pathspec"),
    ],
)

py_binary(
    name = "pyupgrade_py",
    srcs = ["pyupgrade_python.py"],
    main = "pyupgrade_python.py",
    deps = [
        requirement("pyupgrade"),
        requirement("tokenize-rt"),
    ],
)
