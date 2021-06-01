#!/bin/bash
export cc='g++'
meson build --buildtype release -Db_pgo=generate
cd build
meson compile
if [ $! -g 0 ]
then
    break
fi