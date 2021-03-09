
cmake_minimum_required(VERSION 3.14.6)
enable_language(CUDA)
project(bi-ortho_IDR)


add_executable(
biorthoIDR
main.cu
Read_Write_Data.cu
Matrix.cu
Preconditioner.cu
kernels.cu
PIDR_Solver.cu
mmio.cu)
