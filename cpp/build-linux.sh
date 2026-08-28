#!/bin/bash

set -e

TARGET_SOC=rk3588
TARGET_ARCH=aarch64
BUILD_DEMO_NAME=yolov8_seg
export TOOLCHAIN_DIR="${RK3588_TOOLCHAIN_DIR:-/opt/atk-dlrk3588-toolchain}"
export GCC_COMPILER="${TOOLCHAIN_DIR}/bin/aarch64-buildroot-linux-gnu"
echo "$GCC_COMPILER"
# The top-level CMakeLists selects these cross compilers directly.  Codec2 also
# launches a nested, host-native `generate_codebook` build during `make`; its
# ExternalProject inherits the make environment.  Keep the target and host
# compilers separate so that generator is executable on the x86_64 build host.
TARGET_CC=${GCC_COMPILER}-gcc
TARGET_CXX=${GCC_COMPILER}-g++
HOST_CC=${HOST_CC:-/usr/bin/cc}
HOST_CXX=${HOST_CXX:-/usr/bin/c++}
if [[ ! -x "${HOST_CC}" || ! -x "${HOST_CXX}" ]]; then
  echo "Host C/C++ compilers are required for Codec2 code generation" >&2
  exit 1
fi
BUILD_TYPE=Release
CODEC2_CMAKE_ARGS=()
if [[ -n "${ROI_CODEC2_SOURCE_DIR:-}" ]]; then
  CODEC2_CMAKE_ARGS+=("-DROI_CODEC2_SOURCE_DIR=${ROI_CODEC2_SOURCE_DIR}")
fi
case ${TARGET_SOC} in
    rk3588)
        ;;
esac

TARGET_SDK="atk_rknn_${BUILD_DEMO_NAME}_cam"
TARGET_PLATFORM=${TARGET_SOC}_linux
TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
INSTALL_DIR=${ROOT_PWD}/install/${TARGET_PLATFORM}/${TARGET_SDK}
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_SDK}_${TARGET_PLATFORM}_${BUILD_TYPE}
BUILD_DEMO_PATH="./"
echo "==================================="
echo "BUILD_DEMO_NAME=${BUILD_DEMO_NAME}"
echo "BUILD_DEMO_PATH=${BUILD_DEMO_PATH}"
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "ENABLE_ASAN=${ENABLE_ASAN}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "TARGET_CC=${TARGET_CC}"
echo "TARGET_CXX=${TARGET_CXX}"
echo "HOST_CC=${HOST_CC}"
echo "HOST_CXX=${HOST_CXX}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}
cmake ../../${BUILD_DEMO_PATH} \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DTOOLCHAIN_DIR=${TOOLCHAIN_DIR} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    "${CODEC2_CMAKE_ARGS[@]}"
env CC="${HOST_CC}" CXX="${HOST_CXX}" make -j4
env CC="${HOST_CC}" CXX="${HOST_CXX}" make install

# Check if there is a rknn model in the install directory
suffix=".rknn"
shopt -s nullglob
if [ -d "$INSTALL_DIR" ]; then
    files=("$INSTALL_DIR/model/"/*"$suffix")
    shopt -u nullglob

    if [ ${#files[@]} -le 0 ]; then
        echo -e "\e[91mThe RKNN model can not be found in \"$INSTALL_DIR/model\", please check!\e[0m"
    fi
else
    echo -e "\e[91mInstall directory \"$INSTALL_DIR\" does not exist, please check!\e[0m"
fi
