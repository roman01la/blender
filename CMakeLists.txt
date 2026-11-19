cmake_minimum_required(VERSION 3.21)
project(BlenderWeb LANGUAGES CXX)

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure this project with emcmake.")
endif()

add_executable(webgpu-smoke
  blender-platform/GHOST_WebGPUCanvas.cc
  blender-platform/webgpu/webgpu_buffer.cc
  webgpu_smoke.cc
)
target_include_directories(webgpu-smoke PRIVATE blender-platform blender-platform/webgpu)
target_compile_features(webgpu-smoke PRIVATE cxx_std_20)
target_compile_options(webgpu-smoke PRIVATE --use-port=emdawnwebgpu)
target_link_options(webgpu-smoke PRIVATE
  --use-port=emdawnwebgpu
  -sENVIRONMENT=web
  -sEXPORT_ES6=1
)
option(BLENDER_WEB_SMOKE_PTHREADS "Run the smoke test with browser pthreads" OFF)
if(BLENDER_WEB_SMOKE_PTHREADS)
  target_compile_options(webgpu-smoke PRIVATE -pthread)
  target_link_options(webgpu-smoke PRIVATE -pthread -sPTHREAD_POOL_SIZE=1)
endif()
set_target_properties(webgpu-smoke PROPERTIES SUFFIX ".mjs")
configure_file(index.html index.html COPYONLY)
configure_file(favicon.png favicon.png COPYONLY)
