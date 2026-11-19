# SPDX-License-Identifier: GPL-2.0-or-later

# Reuse the Unix library discovery with the Emscripten sysroot.
# Emscripten can compile the NEON probe through an emulated header.
set(SUPPORTS_NEON_BUILD OFF)
# Emscripten supplies allocation at link time. There is no desktop malloc interposer.
set(WITH_TBB_MALLOC_PROXY OFF)
include("${CMAKE_CURRENT_LIST_DIR}/platform_unix.cmake")

if(WITH_PYTHON)
  find_package(BZip2 REQUIRED)
  find_package(expat CONFIG REQUIRED)
  list(APPEND PYTHON_LIBRARIES BZip2::BZip2 ZLIB::ZLIB expat::expat)
  # Emscripten exports the symbols imported by the linked Python side modules.
  set(PYTHON_LINKFLAGS "")
endif()

if(WITH_ALEMBIC)
  find_package(Alembic CONFIG REQUIRED)
  set(ALEMBIC_LIBRARIES Alembic::Alembic)
endif()

if(WITH_OPENVDB)
  find_package(Imath CONFIG REQUIRED)
  find_library(BLENDER_WEB_BLOSC_LIBRARY NAMES blosc REQUIRED)
  list(APPEND OPENVDB_LIBRARIES Imath::Imath "${BLENDER_WEB_BLOSC_LIBRARY}" ZLIB::ZLIB)
endif()

if(WITH_HARU)
  list(APPEND HARU_LIBRARIES PNG::PNG ZLIB::ZLIB)
endif()

if(WITH_CODEC_SNDFILE)
  find_package(SndFile CONFIG REQUIRED)
  set(LIBSNDFILE_LIBRARIES SndFile::sndfile)
endif()

if(WITH_IMAGE_WEBP)
  find_package(WebP CONFIG REQUIRED)
  set(WEBP_LIBRARIES WebP::webpdemux WebP::libwebpmux WebP::webp)
endif()

if(WITH_RUBBERBAND)
  list(APPEND RUBBERBAND_LIBRARIES "${FFTW3_LIBRARY_D}")
endif()

if(WITH_OPENAL)
  list(APPEND OPENAL_INCLUDE_DIR "${EMSCRIPTEN_SYSROOT}/include/AL")
endif()

if(WITH_OPENSUBDIV)
  find_package(TBB CONFIG REQUIRED COMPONENTS tbb)
  find_package(OpenSubdiv CONFIG REQUIRED)
  set(OPENSUBDIV_LIBRARIES OpenSubdiv::osdGPU_static OpenSubdiv::osdCPU_static)
endif()

set(PLATFORM_CFLAGS "-funsigned-char -fno-strict-aliasing -ffp-contract=off")
if(WITH_PYTHON AND (WITH_PYTHON_NUMPY OR WITH_PYTHON_INSTALL_NUMPY))
  add_compile_options(-fPIC)
endif()

# Run Blender's code generators with the target ABI and the build host's filesystem.
if(NOT CMAKE_CROSSCOMPILING_EMULATOR)
  message(FATAL_ERROR "Set CMAKE_CROSSCOMPILING_EMULATOR to a Node.js executable.")
endif()
set(BLENDER_BUILD_TOOL_EMULATOR ${CMAKE_CROSSCOMPILING_EMULATOR})
