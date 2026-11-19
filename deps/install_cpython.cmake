set(archive "libpython${PYTHON_VERSION}.a")
set(embed_archive "libpython${PYTHON_VERSION}-embed.a")

# CPython builds these module support libraries separately from libpython.
# Combine them for consumers of Blender's Python library finder.
set(merge_script "${BINARY_DIR}/merge-python.mri")
file(WRITE "${merge_script}"
  "CREATE ${embed_archive}\n"
  "ADDLIB ${archive}\n"
  "ADDLIB Modules/_decimal/libmpdec/libmpdec.a\n"
  "ADDLIB Modules/_hacl/libHacl_Hash_SHA2.a\n"
  "SAVE\n"
  "END\n"
)
execute_process(
  COMMAND "${WASM_AR}" -M
  WORKING_DIRECTORY "${BINARY_DIR}"
  INPUT_FILE "${merge_script}"
  COMMAND_ERROR_IS_FATAL ANY
)

file(MAKE_DIRECTORY "${SYSROOT}/lib")
file(COPY_FILE "${BINARY_DIR}/${embed_archive}" "${SYSROOT}/lib/${archive}" ONLY_IF_DIFFERENT)
file(COPY "${SOURCE_DIR}/Include/" "${BINARY_DIR}/pyconfig.h"
  DESTINATION "${SYSROOT}/include/python${PYTHON_VERSION}")
file(COPY "${SOURCE_DIR}/Lib/" DESTINATION "${SYSROOT}/lib/python${PYTHON_VERSION}"
  PATTERN "__pycache__" EXCLUDE)

file(STRINGS "${BINARY_DIR}/pybuilddir.txt" sysconfig_directory LIMIT_COUNT 1)
file(GLOB sysconfig_module
  "${BINARY_DIR}/${sysconfig_directory}/_sysconfigdata_*_emscripten_${WASM_ARCHITECTURE}-emscripten.py")
list(LENGTH sysconfig_module sysconfig_count)
if(NOT sysconfig_count EQUAL 1)
  message(FATAL_ERROR "Expected one generated CPython sysconfig module in ${sysconfig_directory}")
endif()
file(COPY ${sysconfig_module} DESTINATION "${SYSROOT}/lib/python${PYTHON_VERSION}")
