cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Deep_RL)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
# Make sure the build directory exists
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/executables)
set(CMAKE_PREFIX_PATH "/home/valentin/Cpp_pack/libtorch" ${CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#add_executable(tensor deepNN_gpt.cpp)
add_executable(Cartpole_DRL CartpoleEnv.h Cartpole_DRL.cpp)
add_executable(Cartpole_new CartpoleEnv.h Cartpole_new.cpp)
#target_link_libraries(tensor "${TORCH_LIBRARIES}")
target_link_libraries(Cartpole_DRL "${TORCH_LIBRARIES}")
target_link_libraries(Cartpole_new "${TORCH_LIBRARIES}")
set_property(TARGET Cartpole_DRL PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET Cartpole_DRL
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:Cartpole_DRL>)
endif (MSVC)