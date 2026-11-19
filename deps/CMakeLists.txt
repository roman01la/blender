cmake_minimum_required(VERSION 3.24)
project(BlenderWebDependencies LANGUAGES C CXX)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure this project with emcmake.")
endif()

include(ExternalProject)
include(../../blender/build_files/build_environment/cmake/versions.cmake)

set(BLENDER_WEB_SYSROOT "${CMAKE_BINARY_DIR}/sysroot" CACHE PATH "Blender's Wasm dependency prefix")
option(BLENDER_WEB_NATIVE_EXCEPTIONS "Build dependencies with the native Wasm exception ABI" OFF)
# These archives also link into the main module that provides symbols to Python extensions.
if(BLENDER_WEB_NATIVE_EXCEPTIONS)
  set(exception_flags "-fwasm-exceptions -sSUPPORT_LONGJMP=wasm")
  set(numpy_native_exceptions true)
else()
  set(exception_flags "-fexceptions")
  set(numpy_native_exceptions false)
endif()
set(dependency_c_flags "${CMAKE_C_FLAGS} -pthread -msimd128 ${exception_flags} -fPIC")
set(dependency_cxx_flags "${CMAKE_CXX_FLAGS} -pthread -msimd128 ${exception_flags} -fPIC")
set(source_mirror "https://projects.blender.org/blender/lib-source/media/branch/main")

function(add_wasm_dependency name version_key)
  cmake_parse_arguments(PARSE_ARGV 2 dependency "" "SOURCE_SUBDIR" "PATCH;OPTIONS;INSTALL")

  set(source_directory_option "")
  if(dependency_SOURCE_SUBDIR)
    set(source_directory_option SOURCE_SUBDIR "${dependency_SOURCE_SUBDIR}")
  endif()
  if(dependency_INSTALL)
    set(install_command INSTALL_COMMAND ${dependency_INSTALL})
  endif()
  if(dependency_PATCH)
    set(patch_command PATCH_COMMAND ${dependency_PATCH})
  endif()

  ExternalProject_Add(${name}
    ${source_directory_option}
    ${install_command}
    ${patch_command}
    URL "${source_mirror}/${${version_key}_FILE}"
    URL_HASH "${${version_key}_HASH_TYPE}=${${version_key}_HASH}"
    TLS_VERIFY ON
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
    LIST_SEPARATOR "|"
    CMAKE_ARGS
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
      "-DCMAKE_C_FLAGS=${dependency_c_flags}"
      "-DCMAKE_CXX_FLAGS=${dependency_cxx_flags}"
      "-DCMAKE_INSTALL_PREFIX=${BLENDER_WEB_SYSROOT}"
      "-DCMAKE_PREFIX_PATH=${BLENDER_WEB_SYSROOT}"
      "-DCMAKE_FIND_ROOT_PATH=${BLENDER_WEB_SYSROOT}"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
      -DBUILD_SHARED_LIBS=OFF
      ${dependency_OPTIONS}
  )
endfunction()

# In zlib 1.3.1, the CMake project declares shared and static targets that
# produce the same archive on Emscripten. Its configure script supports a single static target.
get_filename_component(emscripten_bin "${CMAKE_C_COMPILER}" DIRECTORY)
find_program(wasm_configure emconfigure HINTS "${emscripten_bin}" NO_DEFAULT_PATH REQUIRED)
find_program(wasm_ar emar HINTS "${emscripten_bin}" NO_DEFAULT_PATH REQUIRED)
find_program(wasm_ranlib emranlib HINTS "${emscripten_bin}" NO_DEFAULT_PATH REQUIRED)
find_program(make_program NAMES gmake make REQUIRED)
ExternalProject_Add(zlib
  URL "${source_mirror}/${ZLIB_FILE}"
  URL_HASH "${ZLIB_HASH_TYPE}=${ZLIB_HASH}"
  TLS_VERIFY ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  BUILD_IN_SOURCE ON
  CONFIGURE_COMMAND
    "${CMAKE_COMMAND}" -E env
    "CFLAGS=-O3 ${dependency_c_flags}" "AR=${wasm_ar}" "RANLIB=${wasm_ranlib}"
    "${wasm_configure}" <SOURCE_DIR>/configure --static --uname=Emscripten
    "--prefix=${BLENDER_WEB_SYSROOT}"
  BUILD_COMMAND "${make_program}" libz.a
  INSTALL_COMMAND "${make_program}" install
)

add_wasm_dependency(jpeg JPEG OPTIONS
  "-DCMAKE_INSTALL_LIBDIR=${BLENDER_WEB_SYSROOT}/lib"
  -DENABLE_SHARED=OFF
  -DENABLE_STATIC=ON
  -DWITH_JPEG8=ON
  -DWITH_SIMD=OFF
)
add_wasm_dependency(zstd ZSTD SOURCE_SUBDIR build/cmake OPTIONS
  -DZSTD_BUILD_PROGRAMS=OFF
  -DZSTD_BUILD_SHARED=OFF
  -DZSTD_BUILD_STATIC=ON
  -DZSTD_BUILD_TESTS=OFF
  -DZSTD_MULTITHREAD_SUPPORT=ON
)
add_wasm_dependency(fmt FMT OPTIONS -DFMT_DOC=OFF -DFMT_TEST=OFF)
add_wasm_dependency(glslang SHADERC_GLSLANG OPTIONS
  -DENABLE_OPT=OFF
  -DENABLE_GLSLANG_BINARIES=OFF
  -DGLSLANG_TESTS=OFF
)
add_wasm_dependency(png PNG OPTIONS
  -DPNG_SHARED=OFF
  -DPNG_STATIC=ON
  -DPNG_TESTS=OFF
  -DPNG_TOOLS=OFF
)
ExternalProject_Add_StepDependencies(png configure zlib)

# Brotli 1.0.10 has no install target for Emscripten.
add_wasm_dependency(brotli BROTLI
  OPTIONS -DBROTLI_DISABLE_TESTS=ON
  INSTALL
    "${CMAKE_COMMAND}" -E make_directory "${BLENDER_WEB_SYSROOT}/lib"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
      "<SOURCE_DIR>/c/include/brotli" "${BLENDER_WEB_SYSROOT}/include/brotli"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "<BINARY_DIR>/libbrotlicommon-static.a"
      "<BINARY_DIR>/libbrotlidec-static.a"
      "<BINARY_DIR>/libbrotlienc-static.a"
      "${BLENDER_WEB_SYSROOT}/lib"
)
add_wasm_dependency(freetype FREETYPE OPTIONS
  -DFT_REQUIRE_BROTLI=ON
  -DFT_REQUIRE_ZLIB=ON
  -DFT_DISABLE_BZIP2=ON
  -DFT_DISABLE_HARFBUZZ=ON
  -DFT_DISABLE_PNG=ON
  "-DBROTLIDEC_INCLUDE_DIRS=${BLENDER_WEB_SYSROOT}/include"
  "-DBROTLIDEC_LIBRARIES=${BLENDER_WEB_SYSROOT}/lib/libbrotlidec-static.a|${BLENDER_WEB_SYSROOT}/lib/libbrotlicommon-static.a"
)
ExternalProject_Add_StepDependencies(freetype configure zlib brotli)

ExternalProject_Add(bzip2
  URL "${source_mirror}/${BZIP2_FILE}"
  URL_HASH "${BZIP2_HASH_TYPE}=${BZIP2_HASH}"
  TLS_VERIFY ON
  LOG_DOWNLOAD ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  BUILD_IN_SOURCE ON
  CONFIGURE_COMMAND ""
  BUILD_COMMAND
    "${make_program}" "CC=${CMAKE_C_COMPILER}" "AR=${wasm_ar}"
    "RANLIB=${wasm_ranlib}" "CFLAGS=-O3 ${dependency_c_flags}" libbz2.a
  INSTALL_COMMAND
    "${CMAKE_COMMAND}" -E make_directory "${BLENDER_WEB_SYSROOT}/lib" "${BLENDER_WEB_SYSROOT}/include"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "<SOURCE_DIR>/libbz2.a" "${BLENDER_WEB_SYSROOT}/lib/"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "<SOURCE_DIR>/bzlib.h" "${BLENDER_WEB_SYSROOT}/include/"
)

find_program(host_python "python${PYTHON_SHORT_VERSION}" REQUIRED)
find_program(patch_program patch REQUIRED)
cmake_host_system_information(RESULT cpython_jobs QUERY NUMBER_OF_LOGICAL_CORES)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(wasm_architecture wasm64)
else()
  set(wasm_architecture wasm32)
endif()

ExternalProject_Add(cpython
  URL "${source_mirror}/${PYTHON_FILE}"
  URL_HASH "${PYTHON_HASH_TYPE}=${PYTHON_HASH}"
  TLS_VERIFY ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PATCH_COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/cpython_wasm64_setter.patch"
  CONFIGURE_COMMAND
    "${CMAKE_COMMAND}" -E env
    "CFLAGS=-O3 ${dependency_c_flags} -DPY_CALL_TRAMPOLINE"
    "CPPFLAGS=-I${BLENDER_WEB_SYSROOT}/include"
    "LDFLAGS=-L${BLENDER_WEB_SYSROOT}/lib"
    sh "${CMAKE_CURRENT_SOURCE_DIR}/configure_cpython.sh" "${wasm_configure}"
    "<SOURCE_DIR>" "${wasm_architecture}" "${host_python}" "${BLENDER_WEB_SYSROOT}"
  BUILD_COMMAND
    "${make_program}" "-j${cpython_jobs}" "libpython${PYTHON_SHORT_VERSION}.a" pybuilddir.txt
    "MODULE_ZLIB_CFLAGS=" "MODULE_ZLIB_LDFLAGS=-lz"
    "MODULE_BINASCII_CFLAGS=-DUSE_ZLIB_CRC32" "MODULE_BINASCII_LDFLAGS=-lz"
    "MODULE__BZ2_CFLAGS=" "MODULE__BZ2_LDFLAGS=-lbz2"
  INSTALL_COMMAND
    "${CMAKE_COMMAND}" "-DSOURCE_DIR=<SOURCE_DIR>" "-DBINARY_DIR=<BINARY_DIR>"
    "-DWASM_AR=${wasm_ar}" "-DSYSROOT=${BLENDER_WEB_SYSROOT}"
    "-DPYTHON_VERSION=${PYTHON_SHORT_VERSION}" "-DWASM_ARCHITECTURE=${wasm_architecture}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/install_cpython.cmake"
)
ExternalProject_Add_StepDependencies(cpython configure zlib bzip2
  "${CMAKE_CURRENT_SOURCE_DIR}/configure_cpython.sh")
ExternalProject_Add_StepDependencies(cpython install
  "${CMAKE_CURRENT_SOURCE_DIR}/install_cpython.cmake")

set(BLENDER_WEB_NUMPY_BUILD_PYTHON "${host_python}" CACHE FILEPATH
  "Host Python with Cython installed for building NumPy")
separate_arguments(numpy_c_flags UNIX_COMMAND "${dependency_c_flags}")
separate_arguments(numpy_cxx_flags UNIX_COMMAND "${dependency_cxx_flags}")
list(JOIN numpy_c_flags "', '" numpy_c_flags)
list(JOIN numpy_cxx_flags "', '" numpy_cxx_flags)
find_program(pkg_config_program pkg-config REQUIRED)
configure_file(numpy-cross.ini.in numpy-cross.ini @ONLY)
configure_file(numpy-native.ini.in numpy-native.ini @ONLY)

set(numpy_meson "${BLENDER_WEB_NUMPY_BUILD_PYTHON}" "<SOURCE_DIR>/vendored-meson/meson/meson.py")
set(numpy_site_packages "lib/python${PYTHON_SHORT_VERSION}/site-packages")
ExternalProject_Add(numpy
  URL "${source_mirror}/${NUMPY_FILE}"
  URL_HASH "${NUMPY_HASH_TYPE}=${NUMPY_HASH}"
  TLS_VERIFY ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PATCH_COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/numpy_emscripten.patch"
  CONFIGURE_COMMAND
    "${CMAKE_COMMAND}" -E env
    "BLENDER_WASM_PYTHON_INCLUDE=${BLENDER_WEB_SYSROOT}/include/python${PYTHON_SHORT_VERSION}"
    ${numpy_meson} setup --wipe "<BINARY_DIR>" "<SOURCE_DIR>"
      --cross-file "${CMAKE_CURRENT_BINARY_DIR}/numpy-cross.ini"
      --native-file "${CMAKE_CURRENT_BINARY_DIR}/numpy-native.ini"
      "--prefix=${BLENDER_WEB_SYSROOT}" --libdir lib --buildtype release
      "-Dpython.platlibdir=${numpy_site_packages}" "-Dpython.purelibdir=${numpy_site_packages}"
      -Dblas=openblas -Dlapack=openblas -Dallow-noblas=true
  BUILD_COMMAND ${numpy_meson} compile -C "<BINARY_DIR>"
  INSTALL_COMMAND
    ${numpy_meson} install --no-rebuild --tags runtime,python-runtime,devel -C "<BINARY_DIR>"
)
ExternalProject_Add_StepDependencies(numpy configure cpython
  "${CMAKE_CURRENT_BINARY_DIR}/numpy-cross.ini" "${CMAKE_CURRENT_BINARY_DIR}/numpy-native.ini")

set(blender_dependency_patches
  "${CMAKE_CURRENT_SOURCE_DIR}/../../blender/build_files/build_environment/patches")

add_wasm_dependency(imath IMATH OPTIONS -DBUILD_TESTING=OFF)
add_wasm_dependency(libdeflate DEFLATE OPTIONS
  -DLIBDEFLATE_BUILD_STATIC_LIB=ON
  -DLIBDEFLATE_BUILD_SHARED_LIB=OFF
  -DLIBDEFLATE_BUILD_GZIP=OFF
)
add_wasm_dependency(openjph OPENJPH
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/openjph_table_init_243.diff"
  OPTIONS
    -DOJPH_BUILD_EXECUTABLES=OFF
    -DOJPH_BUILD_TESTS=OFF
    -DOJPH_ENABLE_TIFF_SUPPORT=OFF
)
add_wasm_dependency(openexr OPENEXR
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/openexr_2594.diff"
  OPTIONS
    "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
    -DBUILD_TESTING=OFF
    -DOPENEXR_BUILD_TOOLS=OFF
    -DOPENEXR_BUILD_EXAMPLES=OFF
    -DOPENEXR_INSTALL_TOOLS=OFF
    -DOPENEXR_FORCE_INTERNAL_DEFLATE=OFF
    -DOPENEXR_FORCE_INTERNAL_IMATH=OFF
    -DOPENEXR_FORCE_INTERNAL_OPENJPH=OFF
)
ExternalProject_Add_StepDependencies(openexr configure imath libdeflate openjph)

add_wasm_dependency(openjpeg OPENJPEG OPTIONS -DBUILD_CODEC=OFF -DBUILD_TESTING=OFF)

add_wasm_dependency(fftw FFTW OPTIONS -DBUILD_TESTS=OFF -DENABLE_THREADS=ON)
add_wasm_dependency(fftw_float FFTW OPTIONS
  -DBUILD_TESTS=OFF
  -DENABLE_THREADS=ON
  -DENABLE_FLOAT=ON
)
# The two builds install some common headers and package files.
ExternalProject_Add_StepDependencies(fftw_float install fftw)

add_wasm_dependency(expat EXPAT SOURCE_SUBDIR expat OPTIONS
  -DEXPAT_BUILD_DOCS=OFF
  -DEXPAT_BUILD_EXAMPLES=OFF
  -DEXPAT_BUILD_TESTS=OFF
  -DEXPAT_BUILD_TOOLS=OFF
  -DEXPAT_SHARED_LIBS=OFF
)
ExternalProject_Add_StepDependencies(cpython configure expat)
add_wasm_dependency(yaml_cpp YAMLCPP OPTIONS
  -DYAML_CPP_BUILD_TESTS=OFF
  -DYAML_CPP_BUILD_TOOLS=OFF
  -DYAML_CPP_BUILD_CONTRIB=OFF
)
add_wasm_dependency(pystring PYSTRING
  PATCH "${CMAKE_COMMAND}" -E copy
    "${blender_dependency_patches}/cmakelists_pystring.txt" "<SOURCE_DIR>/CMakeLists.txt"
)
add_wasm_dependency(minizip_ng MINIZIPNG OPTIONS
  -DMZ_FETCH_LIBS=OFF
  -DMZ_LIBCOMP=OFF
  -DMZ_PKCRYPT=OFF
  -DMZ_WZAES=OFF
  -DMZ_OPENSSL=OFF
  -DMZ_SIGNING=OFF
  -DMZ_PPMD=OFF
  -DMZ_LZMA=OFF
  -DMZ_ZSTD=OFF
  -DMZ_BZIP2=OFF
  -DMZ_ICONV=OFF
  -DMZ_ZLIB=ON
  -DMZ_ZLIB_FLAVOR=zlib
)
ExternalProject_Add_StepDependencies(minizip_ng configure zlib)
add_wasm_dependency(opencolorio OPENCOLORIO OPTIONS
  "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
  -DOCIO_INSTALL_EXT_PACKAGES=NONE
  -DOCIO_BUILD_APPS=OFF
  -DOCIO_BUILD_DOCS=OFF
  -DOCIO_BUILD_TESTS=OFF
  -DOCIO_BUILD_GPU_TESTS=OFF
  -DOCIO_BUILD_OPENFX=OFF
  -DOCIO_BUILD_NUKE=OFF
  -DOCIO_BUILD_JAVA=OFF
  -DOCIO_BUILD_PYTHON=OFF
)
ExternalProject_Add_StepDependencies(opencolorio configure imath expat yaml_cpp pystring minizip_ng)

add_wasm_dependency(tbb TBB OPTIONS
  "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
  -DTBB_TEST=OFF
  -DTBB_STRICT=OFF
  -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON
  -DTBBMALLOC_BUILD=ON
  -DTBBMALLOC_PROXY_BUILD=OFF
  -DTBB_ENABLE_IPO=OFF
)

# Open Image Denoise requires 64-bit pointers and runs its compiler on the build host.
if(wasm_architecture STREQUAL "wasm64")
  find_program(BLENDER_WEB_ISPC_EXECUTABLE NAMES ispc NO_CMAKE_FIND_ROOT_PATH REQUIRED)
  add_wasm_dependency(openimagedenoise OIDN
    PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${CMAKE_CURRENT_SOURCE_DIR}/openimagedenoise_wasm64.patch"
    OPTIONS
      "-DISPC_EXECUTABLE=${BLENDER_WEB_ISPC_EXECUTABLE}"
      "-DOIDN_ISPC_WASM_WRAPPER=${CMAKE_CURRENT_SOURCE_DIR}/compile_oidn_ispc.py"
      -DOIDN_APPS=OFF
      -DOIDN_FILTER_RTLIGHTMAP=OFF
      -DOIDN_STATIC_LIB=ON
  )
  ExternalProject_Add_StepDependencies(openimagedenoise configure tbb)
endif()

add_wasm_dependency(eigen EIGEN
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/eigen_tbb_support.diff"
  OPTIONS
    -DBUILD_TESTING=OFF
    -DEIGEN_BUILD_DOC=OFF
    -DEIGEN_BUILD_BLAS=OFF
    -DEIGEN_BUILD_LAPACK=OFF
)
ExternalProject_Add_StepDependencies(eigen configure tbb)

add_wasm_dependency(robinmap ROBINMAP)
add_wasm_dependency(pugixml PUGIXML)
add_wasm_dependency(tiff TIFF
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/tiff_static_dependencies.patch"
  OPTIONS
  -Dtiff-tests=OFF
  -Dtiff-tools=OFF
  -Dtiff-contrib=OFF
  -Dtiff-docs=OFF
  -Dlzma=OFF
  -Djbig=OFF
  -Dzstd=OFF
  -Dwebp=OFF
  -Dlerc=OFF
)
ExternalProject_Add_StepDependencies(tiff configure zlib jpeg libdeflate)
add_wasm_dependency(webp WEBP OPTIONS
  -DWEBP_BUILD_ANIM_UTILS=OFF
  -DWEBP_BUILD_CWEBP=OFF
  -DWEBP_BUILD_DWEBP=OFF
  -DWEBP_BUILD_GIF2WEBP=OFF
  -DWEBP_BUILD_IMG2WEBP=OFF
  -DWEBP_BUILD_VWEBP=OFF
  -DWEBP_BUILD_WEBPINFO=OFF
  -DWEBP_BUILD_WEBPMUX=OFF
  -DWEBP_BUILD_EXTRAS=OFF
)
add_wasm_dependency(aom AOM
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/aom_6d2b7f71b98bfa28e372b1f2d85f137280bdb3de.diff"
  OPTIONS
    -DAOM_TARGET_CPU=generic
    -DENABLE_TESTDATA=OFF
    -DENABLE_TESTS=OFF
    -DENABLE_TOOLS=OFF
    -DENABLE_EXAMPLES=OFF
)
add_wasm_dependency(libheif LIBHEIF
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/libheif_cpp_only.patch"
  OPTIONS
  "-DCMAKE_CXX_FLAGS=${dependency_cxx_flags} -DLIBHEIF_NO_EMBIND"
  -DBUILD_TESTING=OFF
  -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON
  -DWITH_EXAMPLES=OFF
  -DWITH_EXAMPLE_HEIF_VIEW=OFF
  -DWITH_GDK_PIXBUF=OFF
  -DWITH_AOM_ENCODER=ON
  -DWITH_AOM_DECODER=ON
  -DWITH_DAV1D=OFF
  -DWITH_FFMPEG_DECODER=OFF
  -DWITH_JPEG_ENCODER=OFF
  -DWITH_KVAZAAR=OFF
  -DWITH_LIBDE265=OFF
  -DWITH_LIBSHARPYUV=OFF
  -DWITH_OPENJPH_ENCODER=OFF
  -DWITH_OpenH264_DECODER=OFF
  -DWITH_OpenJPEG_DECODER=OFF
  -DWITH_OpenJPEG_ENCODER=OFF
  -DWITH_RAV1E=OFF
  -DWITH_SvtEnc=OFF
  -DWITH_UVG266=OFF
  -DWITH_VVDEC=OFF
  -DWITH_VVENC=OFF
  -DWITH_X265=OFF
  "-DAOM_INCLUDE_DIR=${BLENDER_WEB_SYSROOT}/include"
  "-DAOM_LIBRARY=${BLENDER_WEB_SYSROOT}/lib/libaom.a"
)
ExternalProject_Add_StepDependencies(libheif configure aom tiff jpeg png)

add_wasm_dependency(openimageio OPENIMAGEIO
  PATCH
    "${patch_program}" -p1 -d "<SOURCE_DIR>" -i "${blender_dependency_patches}/openimageio.diff"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${blender_dependency_patches}/openimageio_dds_3d_5133.diff"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${blender_dependency_patches}/openimageio_psd.diff"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${CMAKE_CURRENT_SOURCE_DIR}/openimageio_wasm64_typedefs.patch"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${CMAKE_CURRENT_SOURCE_DIR}/openimageio_emscripten_ustring.patch"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${CMAKE_CURRENT_SOURCE_DIR}/openimageio_static_dependencies.patch"
  OPTIONS
    "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
    "-DOpenImageIO_REQUIRED_DEPS=WebP|libjpeg-turbo|TIFF|OpenEXR|PNG|OpenJPEG|fmt|Robinmap|ZLIB|pugixml|openjph|TBB|Libheif|OpenColorIO"
    -DOpenImageIO_BUILD_MISSING_DEPS=
    -DUSE_SIMD=0
    -DUSE_NUKE=OFF
    -DUSE_OPENVDB=OFF
    -DUSE_FREETYPE=OFF
    -DUSE_DCMTK=OFF
    -DUSE_TBB=ON
    -DUSE_QT=OFF
    -DUSE_PYTHON=OFF
    -DUSE_GIF=OFF
    -DUSE_OPENCV=OFF
    -DUSE_OPENJPEG=ON
    -DUSE_FFMPEG=OFF
    -DUSE_PTEX=OFF
    -DUSE_LIBHEIF=ON
    -DUSE_LIBRAW=OFF
    -DUSE_JXL=OFF
    -DUSE_OPENCOLORIO=ON
    -DUSE_WEBP=ON
    -DUSE_EXTERNAL_PUGIXML=ON
    -DOIIO_BUILD_TOOLS=OFF
    -DOIIO_BUILD_TESTS=OFF
    -DBUILD_TESTING=OFF
)
ExternalProject_Add_StepDependencies(openimageio configure
  openexr opencolorio png jpeg tiff openjpeg fmt robinmap pugixml webp tbb libheif)

add_wasm_dependency(embree EMBREE
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>" -i "${blender_dependency_patches}/embree.diff"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${CMAKE_CURRENT_SOURCE_DIR}/embree_wasm_heap.patch"
  OPTIONS
    -DEMBREE_ISPC_SUPPORT=OFF
    -DEMBREE_SYCL_SUPPORT=OFF
    -DEMBREE_TUTORIALS=OFF
    -DEMBREE_STATIC_LIB=ON
    -DEMBREE_IGNORE_CMAKE_CXX_FLAGS=OFF
    -DEMBREE_MAX_ISA=SSE2
    -DFLAGS_SSE2=-msse2
    -DEMBREE_TASKING_SYSTEM=TBB
    -DEMBREE_RAY_MASK=ON
    -DEMBREE_FILTER_FUNCTION=ON
    -DEMBREE_BACKFACE_CULLING=OFF
    -DEMBREE_BACKFACE_CULLING_CURVES=ON
    -DEMBREE_BACKFACE_CULLING_SPHERES=ON
)
ExternalProject_Add_StepDependencies(embree configure tbb)

add_wasm_dependency(alembic ALEMBIC OPTIONS
  -DUSE_PYALEMBIC=OFF
  -DUSE_ARNOLD=OFF
  -DUSE_MAYA=OFF
  -DUSE_PRMAN=OFF
  -DUSE_HDF5=OFF
  -DUSE_TESTS=OFF
  -DUSE_BINARIES=OFF
  -DALEMBIC_SHARED_LIBS=OFF
)
ExternalProject_Add_StepDependencies(alembic configure imath)

add_wasm_dependency(blosc BLOSC OPTIONS
  "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
  -DBUILD_SHARED=OFF
  -DBUILD_STATIC=ON
  -DBUILD_TESTS=OFF
  -DBUILD_BENCHMARKS=OFF
  -DBUILD_FUZZERS=OFF
  -DCMAKE_C_STANDARD=17
  -DDEACTIVATE_SNAPPY=ON
  -DDEACTIVATE_ZLIB=ON
  -DDEACTIVATE_ZSTD=ON
)

add_wasm_dependency(openvdb OPENVDB
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/openvdb.diff"
  OPTIONS
    "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
    -DBLOSC_USE_STATIC_LIBS=ON
    -DTBB_USE_STATIC_LIBS=ON
    -DOPENVDB_BUILD_UNITTESTS=OFF
    -DOPENVDB_BUILD_NANOVDB=ON
    -DNANOVDB_BUILD_TOOLS=OFF
    -DOPENVDB_CORE_SHARED=OFF
    -DOPENVDB_CORE_STATIC=ON
    -DOPENVDB_BUILD_BINARIES=OFF
    -DOPENVDB_USE_DELAYED_LOADING=OFF
)
ExternalProject_Add_StepDependencies(openvdb configure tbb imath zlib blosc)

add_wasm_dependency(manifold MANIFOLD OPTIONS
  "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
  -DMANIFOLD_JSBIND=OFF
  -DMANIFOLD_CBIND=OFF
  -DMANIFOLD_PYBIND=OFF
  -DMANIFOLD_PAR=ON
  -DMANIFOLD_CROSS_SECTION=OFF
  -DMANIFOLD_EXPORT=OFF
  -DMANIFOLD_TEST=OFF
  -DMANIFOLD_DOWNLOADS=OFF
  -DTRACY_ENABLE=OFF
)
ExternalProject_Add_StepDependencies(manifold configure tbb)

add_wasm_dependency(jolt JOLT SOURCE_SUBDIR Build
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>" -i "${blender_dependency_patches}/jolt_ndebug.diff"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${blender_dependency_patches}/jolt_debug_asset.diff"
  OPTIONS
    "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
    -DJPH_BUILD_SHARED_LIBS=OFF
    -DUSE_WASM_SIMD=ON
    -DCROSS_PLATFORM_DETERMINISTIC=ON
    -DDOUBLE_PRECISION=ON
    -DDEBUG_RENDERER_IN_DEBUG_AND_RELEASE=OFF
    -DPROFILER_IN_DEBUG_AND_RELEASE=OFF
    -DJPH_USE_DX12=OFF
    -DJPH_USE_MTL=OFF
    -DJPH_USE_VK=OFF
    -DTARGET_HELLO_WORLD=OFF
    -DTARGET_PERFORMANCE_TEST=OFF
    -DTARGET_SAMPLES=OFF
    -DTARGET_UNIT_TESTS=OFF
    -DTARGET_VIEWER=OFF
)

add_wasm_dependency(rapidjson RAPIDJSON OPTIONS
  -DRAPIDJSON_BUILD_DOC=OFF
  -DRAPIDJSON_BUILD_EXAMPLES=OFF
  -DRAPIDJSON_BUILD_TESTS=OFF
)

add_wasm_dependency(opentimelineio OPENTIMELINEIO
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/opentimelineio.diff"
  OPTIONS
    -DCMAKE_CXX_STANDARD=17
    -DOTIO_SHARED_LIBS=OFF
    -DOTIO_AUTOMATIC_SUBMODULES=OFF
    -DBUILD_TESTING=OFF
    -DOTIO_CXX_INSTALL=ON
    -DOTIO_PYTHON_INSTALL=OFF
    -DOTIO_DEPENDENCIES_INSTALL=OFF
    -DOTIO_FIND_IMATH=ON
    -DOTIO_FIND_RAPIDJSON=ON
    -DOTIO_FIND_MINIZIP_NG=ON
)
ExternalProject_Add_StepDependencies(opentimelineio configure imath rapidjson minizip_ng zlib)

add_wasm_dependency(haru HARU OPTIONS
  -DMATH_LIB=m
  -DM_LIB=m
  -DLIBHPDF_EXAMPLES=OFF
  -DLIBHPDF_ENABLE_EXCEPTIONS=ON
)
ExternalProject_Add_StepDependencies(haru configure png zlib)

add_wasm_dependency(potrace POTRACE
  PATCH "${CMAKE_COMMAND}" -E copy
    "${blender_dependency_patches}/cmakelists_potrace.txt" "<SOURCE_DIR>/CMakeLists.txt"
)

add_wasm_dependency(abseil ABSEIL OPTIONS
  -DCMAKE_CXX_STANDARD=17
  -DABSL_PROPAGATE_CXX_STD=ON
  -DABSL_BUILD_TESTING=OFF
)

add_wasm_dependency(ceres CERES OPTIONS
  -DBUILD_TESTING=OFF
  -DBUILD_BENCHMARKS=OFF
  -DBUILD_EXAMPLES=OFF
  -DUSE_CUDA=OFF
  -DSUITESPARSE=OFF
  -DLAPACK=OFF
)
ExternalProject_Add_StepDependencies(ceres configure abseil eigen)

add_wasm_dependency(draco DRACO
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/draco_wasm_static.patch"
  OPTIONS
    "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
    -DDRACO_JS_GLUE=OFF
    -DDRACO_TESTS=OFF
)

add_wasm_dependency(meshoptimizer MESHOPTIMIZER OPTIONS
  -DMESHOPT_BUILD_SHARED_LIBS=OFF
)

add_wasm_dependency(ktx KTX
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/ktx_zstd_dep_fix.diff"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${blender_dependency_patches}/ktx_asan.diff"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${CMAKE_CURRENT_SOURCE_DIR}/ktx_wasm_static.patch"
  OPTIONS
    "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
    -DKTX_FEATURE_TOOLS=OFF
    -DKTX_FEATURE_TESTS=OFF
    -DKTX_FEATURE_LOADTEST_APPS=OFF
    -DKTX_FEATURE_ETC_UNPACK=OFF
    -DKTX_FEATURE_GL_UPLOAD=OFF
    -DKTX_FEATURE_VK_UPLOAD=OFF
)
ExternalProject_Add_StepDependencies(ktx configure zstd)

add_wasm_dependency(ogg OGG)

add_wasm_dependency(vorbis VORBIS)
ExternalProject_Add_StepDependencies(vorbis configure ogg)

add_wasm_dependency(flac FLAC OPTIONS
  "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
  -DBUILD_PROGRAMS=OFF
  -DBUILD_EXAMPLES=OFF
  -DBUILD_DOCS=OFF
  -DBUILD_TESTING=OFF
  -DINSTALL_MANPAGES=OFF
)
ExternalProject_Add_StepDependencies(flac configure ogg)

add_wasm_dependency(opus OPUS
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/opus_windows.diff"
  OPTIONS
    "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
    "-DPACKAGE_VERSION=${OPUS_VERSION}"
    -DOPUS_BUILD_PROGRAMS=OFF
    -DOPUS_BUILD_TESTING=OFF
)

ExternalProject_Add(lame
  URL "${source_mirror}/${LAME_FILE}"
  URL_HASH "${LAME_HASH_TYPE}=${LAME_HASH}"
  TLS_VERIFY ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  CONFIGURE_COMMAND
    "${CMAKE_COMMAND}" -E env
    "CC=${CMAKE_C_COMPILER}" "AR=${wasm_ar}" "RANLIB=${wasm_ranlib}"
    "CFLAGS=-O3 ${dependency_c_flags}" "LD=${CMAKE_C_COMPILER}"
    "${wasm_configure}" <SOURCE_DIR>/configure
      --host=none "--prefix=${BLENDER_WEB_SYSROOT}"
      --disable-shared --enable-static --enable-export=full
      --without-vorbis --disable-frontend --disable-mp3x --disable-mp3rtp
      --disable-gtktest
  BUILD_COMMAND "${make_program}"
  INSTALL_COMMAND "${make_program}" install
)

add_wasm_dependency(sndfile SNDFILE
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/sndfile_1045.diff"
  OPTIONS
    -DBUILD_PROGRAMS=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_TESTING=OFF
    -DENABLE_EXTERNAL_LIBS=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_ALSA=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_mpg123=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_Speex=ON
    -DCMAKE_DISABLE_FIND_PACKAGE_SQLite3=ON
    "-DLAME_ROOT=${BLENDER_WEB_SYSROOT}/include"
    "-DMP3LAME_ROOT=${BLENDER_WEB_SYSROOT}/lib"
)
ExternalProject_Add_StepDependencies(sndfile configure ogg vorbis opus flac lame)

find_program(meson_program meson REQUIRED)
separate_arguments(meson_c_flags UNIX_COMMAND "${dependency_c_flags}")
separate_arguments(meson_cxx_flags UNIX_COMMAND "${dependency_cxx_flags}")
list(JOIN meson_c_flags "', '" meson_c_flags)
list(JOIN meson_cxx_flags "', '" meson_cxx_flags)
configure_file(rubberband-cross.ini.in rubberband-cross.ini @ONLY)

ExternalProject_Add(rubberband
  URL "${source_mirror}/${RUBBERBAND_FILE}"
  URL_HASH "${RUBBERBAND_HASH_TYPE}=${RUBBERBAND_HASH}"
  TLS_VERIFY ON
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  PATCH_COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/rubberband_remove_dll_exports.diff"
    COMMAND "${patch_program}" -p1 -d "<SOURCE_DIR>"
      -i "${blender_dependency_patches}/rubberband_missing_cstdlib.diff"
  CONFIGURE_COMMAND
    "${meson_program}" setup --cross-file "${CMAKE_CURRENT_BINARY_DIR}/rubberband-cross.ini"
      "--prefix=${BLENDER_WEB_SYSROOT}" --libdir lib --buildtype release
      -Dauto_features=disabled -Ddefault_library=static -Dfft=fftw
      "<BINARY_DIR>" "<SOURCE_DIR>"
  BUILD_COMMAND "${meson_program}" compile -C "<BINARY_DIR>"
  INSTALL_COMMAND "${meson_program}" install --no-rebuild -C "<BINARY_DIR>"
)
ExternalProject_Add_StepDependencies(rubberband configure fftw)

function(add_wasm_make_dependency name version_key)
  cmake_parse_arguments(PARSE_ARGV 2 dependency "" "" "PATCH;OPTIONS")
  if(dependency_PATCH)
    set(patch_command PATCH_COMMAND ${dependency_PATCH})
  endif()
  ExternalProject_Add(${name}
    ${patch_command}
    URL "${source_mirror}/${${version_key}_FILE}"
    URL_HASH "${${version_key}_HASH_TYPE}=${${version_key}_HASH}"
    TLS_VERIFY ON
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
    CONFIGURE_COMMAND
      "${CMAKE_COMMAND}" -E env
      "CC=${CMAKE_C_COMPILER}" "CXX=${CMAKE_CXX_COMPILER}"
      "AR=${wasm_ar}" "RANLIB=${wasm_ranlib}" "LD=${CMAKE_C_COMPILER}"
      "CFLAGS=-O3 ${dependency_c_flags}" "CXXFLAGS=-O3 ${dependency_cxx_flags}"
      "LDFLAGS=${dependency_c_flags}"
      "PKG_CONFIG=${pkg_config_program}"
      "PKG_CONFIG_LIBDIR=${BLENDER_WEB_SYSROOT}/lib/pkgconfig" "PKG_CONFIG_PATH="
      <SOURCE_DIR>/configure "--prefix=${BLENDER_WEB_SYSROOT}" ${dependency_OPTIONS}
    BUILD_COMMAND "${make_program}" -j4
    INSTALL_COMMAND "${make_program}" install
  )
endfunction()

add_wasm_make_dependency(vpx VPX OPTIONS
  --target=generic-gnu
  --disable-shared --enable-static
  --disable-examples --disable-tools --disable-docs --disable-unit-tests
  --enable-vp8 --enable-vp9 --enable-vp9-highbitdepth --enable-multithread
)

add_wasm_make_dependency(x264 X264
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/x264_emscripten.patch"
  OPTIONS
    "--host=${wasm_architecture}-unknown-emscripten"
    --enable-static --enable-pic --disable-asm --disable-cli
    --disable-lavf --disable-swscale
)

add_wasm_make_dependency(theora THEORA
  PATCH "${patch_program}" -p0 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/theora.diff"
  OPTIONS
    --host=none --disable-shared --enable-static
    --disable-asm --disable-examples
    "--with-ogg=${BLENDER_WEB_SYSROOT}" "--with-vorbis=${BLENDER_WEB_SYSROOT}"
)
ExternalProject_Add_StepDependencies(theora configure ogg vorbis)

set(x265_wasm_options
  "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
  -DENABLE_SHARED=OFF
  -DENABLE_CLI=OFF
  -DENABLE_ASSEMBLY=OFF
)
set(x265_wasm_patch "${patch_program}" -p1 -d "<SOURCE_DIR>"
  -i "${CMAKE_CURRENT_SOURCE_DIR}/x265_cmake_policies.patch")
foreach(bit_depth IN ITEMS 10 12)
  set(main12 OFF)
  if(bit_depth EQUAL 12)
    set(main12 ON)
  endif()
  add_wasm_dependency(x265_${bit_depth} X265 SOURCE_SUBDIR source
    PATCH ${x265_wasm_patch}
    OPTIONS
      ${x265_wasm_options}
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/x265-${bit_depth}"
      -DHIGH_BIT_DEPTH=ON -DEXPORT_C_API=OFF "-DMAIN12=${main12}"
  )
endforeach()
add_wasm_dependency(x265 X265 SOURCE_SUBDIR source
  PATCH ${x265_wasm_patch}
  OPTIONS
    ${x265_wasm_options}
    "-DEXTRA_LIB=${CMAKE_BINARY_DIR}/x265-10/lib/libx265.a|${CMAKE_BINARY_DIR}/x265-12/lib/libx265.a"
    -DLINKED_10BIT=ON -DLINKED_12BIT=ON
)
ExternalProject_Add_StepDependencies(x265 configure x265_10 x265_12)
set(x265_archive "${BLENDER_WEB_SYSROOT}/lib/libx265.a")
ExternalProject_Add_Step(x265 merge_depths
  COMMAND "${CMAKE_COMMAND}" -E rm -f "${x265_archive}.merged"
  COMMAND "${wasm_ar}" qcsL "${x265_archive}.merged" "${x265_archive}"
    "${CMAKE_BINARY_DIR}/x265-10/lib/libx265.a" "${CMAKE_BINARY_DIR}/x265-12/lib/libx265.a"
  COMMAND "${CMAKE_COMMAND}" -E rename "${x265_archive}.merged" "${x265_archive}"
  DEPENDEES install
)

find_program(wasm_nm llvm-nm HINTS "${emscripten_bin}/../bin" NO_DEFAULT_PATH REQUIRED)
add_wasm_make_dependency(ffmpeg FFMPEG
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/ffmpeg.diff"
  OPTIONS
    "--arch=${wasm_architecture}" --target-os=none --enable-cross-compile
    "--cc=${CMAKE_C_COMPILER}" "--cxx=${CMAKE_CXX_COMPILER}"
    "--ar=${wasm_ar}" "--ranlib=${wasm_ranlib}" "--nm=${wasm_nm}"
    "--extra-cflags=${dependency_c_flags} -I${BLENDER_WEB_SYSROOT}/include"
    "--extra-cxxflags=${dependency_cxx_flags}"
    "--extra-ldflags=${dependency_c_flags} -L${BLENDER_WEB_SYSROOT}/lib"
    "--pkg-config=${pkg_config_program}" --pkg-config-flags=--static
    --enable-static --disable-shared
    --disable-autodetect --disable-programs --disable-doc --disable-debug --disable-stripping
    --disable-inline-asm --disable-x86asm --enable-pthreads --enable-simd128
    --enable-gpl --enable-zlib
    --enable-libaom --enable-libopenjpeg --enable-libvpx
    --enable-libx264 --enable-libx265 --enable-libtheora
    --enable-libopus --enable-libvorbis --enable-libmp3lame
)
ExternalProject_Add_StepDependencies(ffmpeg configure
  aom openjpeg vpx x264 x265 theora opus vorbis lame zlib)

add_wasm_dependency(opensubdiv OPENSUBDIV
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${CMAKE_CURRENT_SOURCE_DIR}/opensubdiv_stencil_lookup.patch"
  OPTIONS
  "-DEMSCRIPTEN_SYSTEM_PROCESSOR=${wasm_architecture}"
  -DNO_EXAMPLES=ON -DNO_TUTORIALS=ON -DNO_REGRESSION=ON
  -DNO_TESTS=ON -DNO_GLTESTS=ON -DNO_DOC=ON -DNO_PTEX=ON
  -DNO_OMP=ON -DNO_TBB=OFF
  -DNO_CUDA=ON -DNO_OPENCL=ON -DNO_CLEW=ON -DNO_DX=ON
  -DNO_OPENGL=ON -DNO_METAL=ON -DNO_GLFW=ON -DNO_GLEW=ON
  -DOSD_PATCH_SHADER_SOURCE_GLSL=ON
)
ExternalProject_Add_StepDependencies(opensubdiv configure tbb)

add_wasm_make_dependency(gmp GMP
  PATCH "${patch_program}" -p1 -d "<SOURCE_DIR>"
    -i "${blender_dependency_patches}/gmp.diff"
  OPTIONS
    "--host=${wasm_architecture}-unknown-emscripten"
    --disable-shared --enable-static --disable-assembly --enable-cxx
    "NM=${wasm_nm}"
)

add_wasm_dependency(materialx MATERIALX OPTIONS
  -DMATERIALX_BUILD_SHARED_LIBS=OFF
  -DMATERIALX_BUILD_PYTHON=OFF
  -DMATERIALX_BUILD_JS=OFF
  -DMATERIALX_BUILD_VIEWER=OFF
  -DMATERIALX_BUILD_GRAPH_EDITOR=OFF
  -DMATERIALX_BUILD_TESTS=OFF
  -DMATERIALX_BUILD_RENDER=ON
  -DMATERIALX_BUILD_RENDER_PLATFORMS=OFF
)

find_program(cargo_program NAMES cargo REQUIRED)
set(shader_translator_dir "${CMAKE_CURRENT_SOURCE_DIR}/../shader-translator")
set(shader_translator_target "${CMAKE_CURRENT_BINARY_DIR}/shader-translator-target")
set(shader_translator_wasm
  "${BLENDER_WEB_SYSROOT}/share/blender-web/blender-shader-translator.wasm")
add_custom_command(
  OUTPUT "${shader_translator_wasm}"
  COMMAND "${CMAKE_COMMAND}" -E env --unset=RUSTUP_TOOLCHAIN
    "${cargo_program}" build --locked --release --target wasm32-unknown-unknown
    --target-dir "${shader_translator_target}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${BLENDER_WEB_SYSROOT}/share/blender-web"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${shader_translator_target}/wasm32-unknown-unknown/release/blender_web_shader_translator.wasm"
    "${shader_translator_wasm}"
  WORKING_DIRECTORY "${shader_translator_dir}"
  DEPENDS "${shader_translator_dir}/Cargo.toml" "${shader_translator_dir}/Cargo.lock"
    "${shader_translator_dir}/rust-toolchain.toml" "${shader_translator_dir}/src/lib.rs"
    "${shader_translator_dir}/src/border_sampler.rs"
    "${shader_translator_dir}/src/texture_1d.rs" "${shader_translator_dir}/src/depth_texture.rs"
  VERBATIM
)
add_custom_target(shader_translator ALL DEPENDS "${shader_translator_wasm}")
