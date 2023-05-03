#!/bin/bash

# based on https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7

sudo apt update
sudo apt upgrade

sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall

sudo apt install libjpeg-dev libpng-dev libtiff-dev

sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev 
sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev

sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev

sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd ~

sudo apt-get install libgtk-3-dev

pip install testresources

sudo apt-get install libtbb-dev

sudo apt-get install libatlas-base-dev gfortran

sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# https://github.com/NVlabs/instant-ngp/issues/119
sudo apt install gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10

cd ~/Downloads
# wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip
# wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
unzip opencv.zip
unzip opencv_contrib.zip

# cd opencv-4.7.0
cd opencv
mkdir build
cd build

ls ~/miniconda3/lib/python3.10/site-packages/
ls ~/miniconda3/bin/python
# ls ~/Downloads/opencv_contrib-4.7.0/modules
ls ~/Downloads/opencv_contrib/modules
ls ~/miniconda3/include

# cmake -D CMAKE_BUILD_TYPE=RELEASE \
# -D CMAKE_INSTALL_PREFIX=/usr/local \
# -D WITH_TBB=ON \
# -D ENABLE_FAST_MATH=1 \
# -D CUDA_FAST_MATH=1 \
# -D WITH_CUBLAS=1 \
# -D WITH_CUDA=ON \
# -D BUILD_opencv_cudacodec=OFF \
# -D WITH_CUDNN=ON \
# -D OPENCV_DNN_CUDA=ON \
# -D CUDA_ARCH_BIN=7.5 \
# -D WITH_V4L=ON \
# -D WITH_QT=OFF \
# -D WITH_OPENGL=ON \
# -D WITH_GSTREAMER=ON \
# -D OPENCV_GENERATE_PKGCONFIG=ON \
# -D OPENCV_PC_FILE_NAME=opencv.pc \
# -D OPENCV_ENABLE_NONFREE=ON \
# -D OPENCV_PYTHON3_INSTALL_PATH=~/miniconda3/lib/python3.10/site-packages/ \
# -D PYTHON_EXECUTABLE=~/miniconda3/bin/python \
# -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib-4.7.0/modules \
# -D INSTALL_PYTHON_EXAMPLES=OFF \
# -D INSTALL_C_EXAMPLES=OFF \
# -D CUDAToolkit_ROOT=~/miniconda3/bin/nvcc \
# -D CUDNN_LIBRARY=~/miniconda3/lib/libcudnn.so.8 \
# -D CUDNN_VERSION=8.2.1 \
# -D CUDNN_INCLUDE_DIR=~/miniconda3/include \
# -D CUDA_ARCH_PTX="" \
# -D CUDA_ARCH_BIN=7.5 \
# -D BUILD_EXAMPLES=OFF ..

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=7.5 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_PYTHON3_INSTALL_PATH=~/miniconda3/lib/python3.10/site-packages/ \
-D PYTHON_EXECUTABLE=~/miniconda3/bin/python \
-D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D CUDAToolkit_ROOT=~/miniconda3/bin/nvcc \
-D CUDNN_LIBRARY=~/miniconda3/lib/libcudnn.so.8 \
-D CUDNN_VERSION=8.2.1 \
-D CUDNN_INCLUDE_DIR=~/miniconda3/include \
-D CUDA_ARCH_PTX="" \
-D CUDA_ARCH_BIN=7.5 \
-D BUILD_EXAMPLES=OFF ..

nproc
make -j32
sudo make install


ln -s /usr/local/lib/python3.6/site-packages/cv2 ~/miniconda3/lib/python3.10/site-packages/cv2
