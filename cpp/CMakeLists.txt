cmake_minimum_required(VERSION 3.15)

set(TOOLCHAIN_DIR "/opt/atk-dlrk3588-toolchain" CACHE PATH
    "RK3588 Buildroot cross-toolchain root")
if (NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu-gcc")
endif()
if (NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu-g++")
endif()
if (NOT DEFINED CMAKE_SYSROOT)
    set(CMAKE_SYSROOT "${TOOLCHAIN_DIR}/aarch64-buildroot-linux-gnu/sysroot")
endif()

project(rknn_yolov8_seg_cam)

set(SYSROOT ${CMAKE_SYSROOT}/usr/include)
set(CMAKE_CXX_STANDARD 11)
add_definitions(-g -ggdb -gdwarf -funwind-tables -rdynamic)
add_definitions(-Wno-write-strings -Wno-return-type)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

#opencv
include_directories(${SYSROOT})
include_directories(${SYSROOT}/opencv4)
set(OPENCV_LIBS opencv_core opencv_flann opencv_videoio opencv_video opencv_highgui opencv_imgcodecs opencv_imgproc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# The executable is installed directly beside lib/, not inside a bin/ folder.
# Keep deployed RKNN/RGA/MPP libraries resolvable without an LD_LIBRARY_PATH.
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

# The vendor SDK currently ships a GLIBC_2.38 MPP runtime, while the target
# ATK-DLRK3588 Buildroot image provides glibc 2.37 and its own compatible
# librockchip_mpp.so.1.  Link against the SDK for headers and symbols, but use
# the tested board runtime by default.  Set this explicitly only for target
# images known to use the SDK's glibc level or newer.
option(MPP_BUNDLE_RUNTIME "Install the SDK Rockchip MPP runtime beside the application" OFF)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

#dma
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/dma)
#drm
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/drm)


set(postprocess_file rknpu2/postprocess.cc)
set(yolov8_seg_file rknpu2/yolov8_seg.cc)
file(GLOB_RECURSE PIPELINE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/capture/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/audio/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/inference/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/preview/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/roi/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/encoder/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/transport/*.cc
)

# The production sender must use RK3588 MPP. Do not silently fall back to x265
# or a software encoder when an SDK/sysroot is incomplete.
find_path(MPP_INCLUDE_DIR rk_mpi.h
    PATHS
        ${CMAKE_SYSROOT}/usr/include
        ${CMAKE_SYSROOT}/usr/include/rockchip
        ${CMAKE_SYSROOT}/usr/include/mpp
)
find_path(MPP_VENC_INCLUDE_DIR rk_venc_cmd.h
    PATHS
        ${CMAKE_SYSROOT}/usr/include
        ${CMAKE_SYSROOT}/usr/include/rockchip
        ${CMAKE_SYSROOT}/usr/include/mpp
)
find_library(MPP_LIBRARY NAMES rockchip_mpp mpp
    PATHS
        ${CMAKE_SYSROOT}/usr/lib
        ${CMAKE_SYSROOT}/usr/lib64
        ${CMAKE_SYSROOT}/lib
)
if (NOT MPP_INCLUDE_DIR OR NOT MPP_VENC_INCLUDE_DIR OR NOT MPP_LIBRARY)
    message(FATAL_ERROR "Rockchip MPP (rk_mpi.h, rk_venc_cmd.h, and librockchip_mpp/libmpp) is required for the H.265 sender")
endif()

# Codec2 is built as a bundled shared library so the Buildroot board does not
# need a preinstalled libcodec2 package.  A local source checkout can be
# supplied for offline builds; otherwise CMake fetches the pinned 1.2.0 source
# archive once into its build cache.  Keep this after the MPP check so an
# incomplete video toolchain fails immediately without a needless download.
set(ROI_CODEC2_SOURCE_DIR "" CACHE PATH "Optional local Codec2 1.2.0 source directory")
set(LPCNET OFF CACHE BOOL "Disable optional LPCNet in the Codec2 dependency" FORCE)
set(UNITTEST OFF CACHE BOOL "Do not build Codec2 unit tests with the sender" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build Codec2 as a deployable shared library" FORCE)
if (ROI_CODEC2_SOURCE_DIR)
    set(ROI_CODEC2_BINARY_DIR "${CMAKE_BINARY_DIR}/codec2-build")
    add_subdirectory("${ROI_CODEC2_SOURCE_DIR}" "${ROI_CODEC2_BINARY_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(roi_codec2
        URL "https://github.com/drowe67/codec2/archive/refs/tags/1.2.0.tar.gz"
        URL_HASH "SHA256=cbccae52b2c2ecc5d2757e407da567eb681241ff8dadce39d779a7219dbcf449"
    )
    FetchContent_MakeAvailable(roi_codec2)
    set(ROI_CODEC2_BINARY_DIR "${roi_codec2_BINARY_DIR}")
endif()
if (NOT TARGET codec2)
    message(FATAL_ERROR "Codec2 1.2.0 is required for the audio RTP sender")
endif()


add_executable(${PROJECT_NAME}
    main.cc
    glibc_compat.c
    ${postprocess_file}
    ${yolov8_seg_file}
    ${PIPELINE_SRCS}
)


target_link_libraries(${PROJECT_NAME}
    fileutils
    imageutils
    imagedrawing
    ${OPENCV_LIBS}    
    ${LIBRKNNRT}
    ${MPP_LIBRARY}
    codec2
)


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${MPP_INCLUDE_DIR}
    ${MPP_VENC_INCLUDE_DIR}
    # Codec2 generates codec2/version.h in its binary directory.  Its
    # upstream target exposes CMAKE_BINARY_DIR instead, which is the parent
    # project's build directory when it is included as a subdirectory.
    ${ROI_CODEC2_BINARY_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_ROCKCHIP_MPP=1)

get_filename_component(MPP_LIBRARY_REALPATH "${MPP_LIBRARY}" REALPATH)
if (NOT CMAKE_READELF)
    find_program(CMAKE_READELF NAMES aarch64-buildroot-linux-gnu-readelf readelf
        PATHS "${TOOLCHAIN_DIR}/bin")
endif()
if (NOT CMAKE_READELF)
    message(FATAL_ERROR "Cannot locate readelf to determine the Rockchip MPP runtime SONAME")
endif()
execute_process(
    COMMAND "${CMAKE_READELF}" -d "${MPP_LIBRARY_REALPATH}"
    RESULT_VARIABLE MPP_READELF_RESULT
    OUTPUT_VARIABLE MPP_READELF_OUTPUT
    ERROR_VARIABLE MPP_READELF_ERROR
)
string(REGEX MATCH "SONAME[^\n]*\\[([^]]+)\\]" MPP_SONAME_MATCH "${MPP_READELF_OUTPUT}")
set(MPP_LIBRARY_SONAME "${CMAKE_MATCH_1}")
if (NOT MPP_READELF_RESULT EQUAL 0 OR NOT MPP_LIBRARY_SONAME)
    message(FATAL_ERROR "Cannot determine the Rockchip MPP runtime SONAME: ${MPP_READELF_ERROR}")
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${LIBRKNNRT} DESTINATION lib)
if (MPP_BUNDLE_RUNTIME)
    install(FILES ${MPP_LIBRARY_REALPATH} DESTINATION lib RENAME "${MPP_LIBRARY_SONAME}")
else()
    message(STATUS "Using the target image's ${MPP_LIBRARY_SONAME}; SDK MPP runtime will not be installed")
endif()
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)

file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)
