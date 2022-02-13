wget http://mirrors.ustc.edu.cn/gnu/libc/glibc-2.18.tar.gz

tar -zxvf glibc-2.18.tar.gz
cd glibc-2.18
mkdir build && cd build
../configure --prefix=/opt/glibc-2.18
make -j4 && make install

export GLIBC_PATH=/opt/glibc-2.18/lib
strings /usr/lib64/libc.so.6 | grep GLIBC_2.18