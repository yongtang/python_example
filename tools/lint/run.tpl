#!/usr/bin/env bash

RUN_PATH=@@RUN_PATH@@
RUN_ARGS=@@RUN_ARGS@@

RUN_PATH=$(readlink "$RUN_PATH")

option=$(basename $RUN_PATH)

if [[ $option == 'clang-binary' ]]; then
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in \
        $( \
            find foo -type f \
                \( -name '*.cc' -o -name '*.h' -o -name '*.cpp' \) \
        ) ; do \
        echo $RUN_PATH $RUN_ARGS "$i" ; \
        $RUN_PATH $RUN_ARGS "$i" ; \
    done \
)
else
( \
    cd "$BUILD_WORKSPACE_DIRECTORY" && \
    for i in \
        $( \
            find foo tests -type f \
                \( -name '*.py' \) \
        ) ; do \
        echo $RUN_PATH $RUN_ARGS "$i" ; \
        $RUN_PATH $RUN_ARGS "$i" ; \
    done \
)
fi
