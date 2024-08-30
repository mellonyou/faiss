# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_TEST_SRC
  test_callback.cpp
)

add_executable(faiss_test ${FAISS_TEST_SRC})

if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512" AND NOT FAISS_OPT_LEVEL STREQUAL "sve")
  target_link_libraries(faiss_test PRIVATE faiss)
endif()

if(FAISS_OPT_LEVEL STREQUAL "avx2")
  if(NOT WIN32)
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
  else()
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  endif()
  target_link_libraries(faiss_test PRIVATE faiss_avx2)
endif()

if(FAISS_OPT_LEVEL STREQUAL "avx512")
  if(NOT WIN32)
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw>)
  else()
    target_compile_options(faiss_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  endif()
  target_link_libraries(faiss_test PRIVATE faiss_avx512)
endif()

if(FAISS_OPT_LEVEL STREQUAL "sve")
  if(NOT WIN32)
    if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=native")
      # Do nothing, expect SVE to be enabled by -march=native
    elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
      # Add +sve
      target_compile_options(faiss_test PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:${CMAKE_MATCH_2}+sve>)
    elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} " MATCHES "(^| )-march=armv")
      # No valid -march, so specify -march=armv8-a+sve as the default
      target_compile_options(faiss_test PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-march=armv8-a+sve>)
    endif()
    if("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=native")
      # Do nothing, expect SVE to be enabled by -march=native
    elseif("${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )(-march=armv[0-9]+(\\.[1-9]+)?-[^+ ](\\+[^+$ ]+)*)")
      # Add +sve
      target_compile_options(faiss_test PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:${CMAKE_MATCH_2}+sve>)
    elseif(NOT "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} " MATCHES "(^| )-march=armv")
      # No valid -march, so specify -march=armv8-a+sve as the default
      target_compile_options(faiss_test PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-march=armv8-a+sve>)
    endif()
  else()
    # TODO: support Windows
  endif()
  target_link_libraries(faiss_test PRIVATE faiss_sve)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850 # release-1.12.1
  OVERRIDE_FIND_PACKAGE)
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

if(NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config.cmake
   AND NOT EXISTS ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/GTestConfig.cmake)
  file(
    WRITE ${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/gtest-config.cmake
    [=[
include(CMakeFindDependencyMacro)
find_dependency(googletest)
if(NOT TARGET GTest::GTest)
  add_library(GTest::GTest INTERFACE IMPORTED)
  target_link_libraries(GTest::GTest INTERFACE GTest::gtest)
endif()
if(NOT TARGET GTest::Main)
  add_library(GTest::Main INTERFACE IMPORTED)
  target_link_libraries(GTest::Main INTERFACE GTest::gtest_main)
endif()
]=])
endif()

find_package(OpenMP REQUIRED)
find_package(GTest CONFIG REQUIRED)

target_link_libraries(faiss_test PRIVATE
  OpenMP::OpenMP_CXX
  GTest::gtest_main
  $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft>
  $<$<BOOL:${FAISS_ENABLE_ROCM}>:hip::host>
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(faiss_test)
