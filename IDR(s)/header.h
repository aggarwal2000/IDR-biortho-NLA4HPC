/*!
\file header.h
\brief important macros for GPU kernels
*/
# pragma once
//! defines the number of threads per block
#define THREADS_PER_BLOCK 256 

//! size of warp(as provided by CUDA )
#define WARP_SIZE 32


#define FULL_MASK 0xffffffff
