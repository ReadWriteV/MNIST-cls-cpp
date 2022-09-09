cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist_cls_cpp)

find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_library(mnist_cls_model model.cpp)
add_executable(train train.cpp)
add_executable(test test.cpp)

target_link_libraries(mnist_cls_model PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(train mnist_cls_model Boost::program_options)
target_link_libraries(test mnist_cls_model Boost::program_options)
set_property(TARGET mnist_cls_model train test PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET ${TARGET_NAME}
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:${TARGET_NAME}>)
# endif (MSVC)
