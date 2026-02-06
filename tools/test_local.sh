# meson build -Denable-opencl=true

ninja -C build install
./build/test/unittest/unittest_opencl_kernels_int4