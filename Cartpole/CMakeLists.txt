cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(randomactions)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
# Make sure the build directory exists
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/executables)
set(CMAKE_PREFIX_PATH "/home/valentin/Cpp_pack/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#add_executable(tensor deepNN_gpt.cpp)
add_executable(randomactions randomactions.cpp)
add_executable(deepl_actions deepl_actions.cpp)
add_executable(slow_build_up slow_build_up.cpp)
add_executable(trying_new trying_new.cpp)
#target_link_libraries(tensor "${TORCH_LIBRARIES}")
target_link_libraries(randomactions "${TORCH_LIBRARIES}")
target_link_libraries(deepl_actions "${TORCH_LIBRARIES}")
target_link_libraries(slow_build_up "${TORCH_LIBRARIES}")
target_link_libraries(trying_new "${TORCH_LIBRARIES}")
set_property(TARGET randomactions PROPERTY CXX_STANDARD 14)
set_property(TARGET deepl_actions PROPERTY CXX_STANDARD 14)
set_property(TARGET trying_new PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET tensor
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:tensor>)
endif (MSVC)