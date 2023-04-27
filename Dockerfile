FROM nvcr.io/nvidia/pytorch:22.08-py3

ARG DEBIAN_FRONTEND=noninteractive

ARG OPENCV_REPO=https://github.com/opencv/opencv.git
ARG OPENCV_VERSION=4.0.0
ARG OPENCV_CONTRIB_REPO=https://github.com/opencv/opencv_contrib.git
ARG OPENCV_CONTRIB_VERSION=4.0.0

# Install build dependencies
RUN apt-get update \
    # Install build tools
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential \
        cmake \
        gdb \
        ninja-build \
    # Install opencv build dependencies
    && apt-get install -y --no-install-recommends \
        libeigen3-dev \
        python3-dev \
        python3-numpy \
        python3-matplotlib \
    # Install rocker dependecies
    && apt-get install -y --no-install-recommends \
        libgtk2.0-dev \
    # Intall dev packages
    && apt-get install -y --no-install-recommends \
        valgrind \
        python3-pip \
        python3-tk \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone $OPENCV_REPO opencv -b $OPENCV_VERSION \
    && git clone $OPENCV_CONTRIB_REPO opencv_contrib -b $OPENCV_CONTRIB_VERSION \
    && mkdir -p opencv/build \
    && cd opencv/build \
    && cmake -GNinja \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DWITH_EIGEN=ON \
        -DWITH_CUDA=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DPYTHON3_EXECUTABLE=/usr/bin/python3 \
        -DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include/ \
        -DBUILD_EXAMPLES=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        .. \
    && ninja install \
    && ldconfig \
    && cd ../.. \
    && rm -rf opencv opencv_contrib

# Setting up Pangolin lib
# Installing dependecies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \ 
    libglew-dev \
    libwayland-dev \
    libxkbcommon-dev \
    wayland-protocols \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/stevenlovegrove/Pangolin.git Pangolin -b v0.6 \
    && mkdir -p Pangolin/build \
    && cd Pangolin/build \ 
    && cmake .. \
    && cmake --build .

COPY . /root/kltvo/

RUN cd /root/kltvo \
    && mkdir -p build/ \
    && cd  build && cmake .. \
    && make

# Benchmark tools
RUN pip3 install evo --upgrade --no-binary evo \
    && pip3 install pandas seaborn

WORKDIR /root/kltvo
CMD ["bash"]