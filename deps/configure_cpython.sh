#!/bin/sh
set -eu

emconfigure=$1
source_dir=$2
architecture=$3
host_python=$4
install_prefix=$5

# The platform-triplet probe uses the preprocessor without CFLAGS.
if [ "$architecture" = wasm64 ]; then
  export CPPFLAGS="${CPPFLAGS:-} -m64"
fi

rm -f pybuilddir.txt
export CONFIG_SITE="$source_dir/Tools/wasm/config.site-wasm32-emscripten"
exec "$emconfigure" "$source_dir/configure" \
  "--build=$("$source_dir/config.guess")" "--host=$architecture-unknown-emscripten" \
  --with-emscripten-target=browser --enable-wasm-pthreads --disable-shared \
  --with-system-expat \
  "--with-build-python=$host_python" "--prefix=$install_prefix"
