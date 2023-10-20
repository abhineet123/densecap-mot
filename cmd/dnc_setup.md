<!-- MarkdownTOC -->

- [virtualenv](#virtualen_v_)
  - [x99       @ virtualenv](#x99___virtualenv_)
  - [e5g       @ virtualenv](#e5g___virtualenv_)
  - [windows       @ virtualenv](#windows___virtualenv_)
- [install](#install_)
- [torch](#torch_)
- [extract_feature](#extract_feature_)
  - [opencv_from_source       @ extract_feature](#opencv_from_source___extract_featur_e_)
    - [x99_/CUDA_11.3/rtx3060       @ opencv_from_source/extract_feature](#x99__cuda_11_3_rtx3060___opencv_from_source_extract_feature_)
    - [grs/ubuntu22.04       @ opencv_from_source/extract_feature](#grs_ubuntu22_04___opencv_from_source_extract_feature_)
  - [caffe-action       @ extract_feature](#caffe_action___extract_featur_e_)
    - [cmake       @ caffe-action/extract_feature](#cmake___caffe_action_extract_feature_)
  - [caffe-source       @ extract_feature](#caffe_source___extract_featur_e_)
  - [denseflow       @ extract_feature](#denseflow___extract_featur_e_)
    - [orca       @ denseflow/extract_feature](#orca___denseflow_extract_featur_e_)
    - [x99       @ denseflow/extract_feature](#x99___denseflow_extract_featur_e_)
- [x99 dpkg mess](#x99_dpkg_mess_)
    - [caffe       @ x99_dpkg_mess/](#caffe___x99_dpkg_mess_)
- [bugs](#bug_s_)

<!-- /MarkdownTOC -->

<a id="virtualen_v_"></a>
# virtualenv
mkvirtualenv dnc
<a id="x99___virtualenv_"></a>
## x99       @ virtualenv-->dnc_setup
mkvirtualenv -p python3.6 dnc<a id="x99___virtualenv_"></a>
<a id="e5g___virtualenv_"></a>
## e5g       @ virtualenv-->dnc_setup
mkvirtualenv -p python3.7 dnc

nano ~/.bashrc
alias dnc='workon dnc'
source ~/.bashrc

<a id="windows___virtualenv_"></a>
## windows       @ virtualenv-->dnc_setup
virtualenv dnc
dnc\Scripts\activate

<a id="install_"></a>
# install
python -m pip install pip==20.0.0
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

<a id="torch_"></a>
# torch
python -m pip install torch==0.4.0+cu113
python -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

<a id="extract_feature_"></a>
# extract_feature
python -m pip install protobuf easydict youtube-dl Flask paramparse scikit-image

python -m pip install opencv-python
python -m pip uninstall opencv-python

<a id="opencv_from_source___extract_featur_e_"></a>
## opencv_from_source       @ extract_feature-->dnc_setup
3.4.1 from acamp install.md
```
wget https://github.com/opencv/opencv/archive/3.4.1.zip
unzip 3.4.1.zip
rm 3.4.1.zip
wget https://github.com/opencv/opencv_contrib/archive/3.4.1.zip
unzip 3.4.1.zip
rm 3.4.1.zip
cd opencv-3.4.1
mkdir build
cd build
```
<a id="x99__cuda_11_3_rtx3060___opencv_from_source_extract_feature_"></a>
### x99_/CUDA_11.3/rtx3060       @ opencv_from_source/extract_feature-->dnc_setup
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D BUILD_opencv_cudacodec=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.1/modules \
      -D CUDA_ARCH_BIN=8.6 \
      -D WITH_CUDA=ON ..  
make -j16 --std=c++14
```

<a id="grs_ubuntu22_04___opencv_from_source_extract_feature_"></a>
### grs/ubuntu22.04       @ opencv_from_source/extract_feature-->dnc_setup
__ISO C++17 does not allow dynamic exception specifications__
add in CMakeLists.txt after project() line
```
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```
__eliminate the great gluttony of warnings__
remove
```
add_extra_compiler_option(-Wall)
```
from line 74 of `cmake/OpenCVCompilerOptions.cmake`
along with all other -W crap from there

__stop on first error__
add this to the above file:
```
add_extra_compiler_option(-Wfatal-errors)
```
__cmake/OpenCVDetectCUDA.cmake__
replace with the version in lib/dense_flow/cmake/opencv
or edit thus:

line 23
```
  if(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
    ocv_list_filterout(CUDA_nppi_LIBRARY "nppicom")
    ocv_list_filterout(CUDA_npp_LIBRARY "nppicom")
  endif()
```
line 106:
```
      if(${CUDA_VERSION} VERSION_LESS "9.0")
        set(__cuda_arch_bin "2.0 3.0 3.5 3.7 5.0 5.2 6.0 6.1")
      elseif(CUDA_VERSION VERSION_GREATER_EQUAL "10.2")
        set(__cuda_arch_bin "5.0 5.2 6.0 6.1 7.0 8.6")
      else()
        set(__cuda_arch_bin "3.0 3.5 3.7 5.0 5.2 6.0 6.1 7.0")
      endif()
```
<a id="caffe_action___extract_featur_e_"></a>
## caffe-action       @ extract_feature-->dnc_setup
```
sudo apt-get install --no-install-recommends libboost-all-dev==1.58
sudo apt-get install libatlas-base-dev

sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```
cd feature_extract/lib/caffe-action
mkdir build 
cd build

OpenCV_DIR=~/opencv-3.4.1/build cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
make -j16

<a id="cmake___caffe_action_extract_feature_"></a>
### cmake       @ caffe-action/extract_feature-->dnc_setup
upgrade cmake to >=3.12 to resolve the annoying CUDA_cublas_device_LIBRARY error
https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line

sudo apt remove --purge cmake
sudo apt remove --purge --auto-remove cmake

sudo apt update && \
sudo apt install -y software-properties-common lsb-release && \
sudo apt clean all

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null

sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

sudo apt update
sudo apt install cmake

<a id="caffe_source___extract_featur_e_"></a>
## caffe-source       @ extract_feature-->dnc_setup
__not needed__

https://caffe.berkeleyvision.org/installation.html#compilation
https://anidh.medium.com/install-caffe-on-ubuntu-with-cuda-6d0da9e8f860

pip3 install protobuf

sudo apt-get install libopencv-dev
sudo apt-get install libopencv-contrib-dev
sudo apt-get install libopencv-features2d-dev

sudo apt-get install python3.6-dev
sudo apt-get remove libopencv-dev
sudo apt-get remove libopencv-contrib-dev
sudo apt-get remove libopencv-features2d-dev

git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
nano Makefile.config

/usr/local/lib/python3.6/dist-packages/numpy/core/include

make all -j16
make pycaffe -j16

make clean

<a id="denseflow___extract_featur_e_"></a>
## denseflow       @ extract_feature-->dnc_setup
sudo apt install libzip-dev
sudo apt install cmake

sudo ln -s /usr/lib/x86_64-linux-gnu/libboost_python3-py36.so /usr/lib/libboost_python36.so

cd feature_extract/lib/dense_flow
mkdir build 
cd build

cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF -DCMAKE_C_COMPILER=/usr/bin/gcc
make -j16

<a id="orca___denseflow_extract_featur_e_"></a>
### orca       @ denseflow/extract_feature-->dnc_setup
/usr/lib/x86_64-linux-gnu/libboost_python-py27.so.1.58.0

<a id="x99___denseflow_extract_featur_e_"></a>
### x99       @ denseflow/extract_feature-->dnc_setup
cp /home/abhineet/ubuntu16/usr/local/cuda-9.0/include/dynlink_nvcuvid.h /usr/local/cuda-10.2/include/

cp /home/abhineet/ubuntu16/usr/local/cuda-9.0/include/dynlink_cuviddec.h /usr/local/cuda-10.2/include/

cp /home/abhineet/ubuntu16/usr/local/cuda-9.0/include/dynlink_cuda.h /usr/local/cuda-10.2/include/

cp /home/abhineet/ubuntu16/usr/local/cuda-9.0/include/dynlink_cuda_cuda.h /usr/local/cuda-10.2/include/


<a id="x99_dpkg_mess_"></a>
# x99 dpkg mess
dpkg-divert: error: rename involves overwriting '/usr/share/dict/words.pre-dictionaries-common' with
  different file '/usr/share/dict/words', not allowed

sudo mv /usr/share/dict/words.pre-dictionaries-common /usr/share/dict/words.pre-dictionaries-common.bak

sudo apt-get -o DPkg::Options::="--force-confnew" -y upgrade
sudo apt-get -o DPkg::Options::="--force-confnew" -y dist-upgrade

sudo dpkg --force-confdef --force-confnew --configure -a

ln -sf /usr/lib/x86_64-linux-gnu/libboost_python38.so /usr/lib/x86_64-linux-gnu/libboost_python3.so

for package in $(apt-get upgrade 2>&1 |\
                 grep "warning: files list file for package '" |\
                 grep -Po "[^'\n ]+'" | grep -Po "[^']+"); do
    apt-get install --reinstall "$package";
done

<a id="caffe___x99_dpkg_mess_"></a>
### caffe       @ x99_dpkg_mess/-->dnc_setup
apt-cache search gflags
sudo apt-get install libgflags2.2 libgflags-dev


apt-cache search glog
sudo apt-get install libgoogle-glog0v5 libgoogle-glog-dev


apt -y install gcc-8 g++-8
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

sudo cp /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/local/lib/libhdf5.so
sudo cp /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so /usr/local/lib/libhdf5_hl.so

sudo apt remove caffe-cuda

sudo apt remove caffe-cpu

<a id="bug_s_"></a>
# bugs
`dist.init_process_group timeout`: set world_size to 1; world_size > 1 is intended for multiple servers working together

also, apparently changing backend to nccl from gloo seems to help a bit'

`multiprocessing oserror errno 24 too many open files`
add
`torch.multiprocessing.set_sharing_strategy('file_system')`
at the beginning of the script

turns out that you should not pass stuff like `world_size` and `rank` to `init_process_group` at all as suggested here:
https://pytorch.org/docs/stable/distributed.html#distributed-launch


`ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).`
`RuntimeError: unable to write to file </torch_1446540_2397425596>`
https://landoflinux.com/linux_shared_memory_configure.html#:~:text=To%20change%20the%20size%20of,%2Fdev%2Fshm%22%20command.
sudo nano /etc/fstab
tmpfs /dev/shm tmpfs defaults,size=64G,nodev,nosuid 0 0
sudo mount -o remount /dev/shm
df -h /dev/shm
































































































