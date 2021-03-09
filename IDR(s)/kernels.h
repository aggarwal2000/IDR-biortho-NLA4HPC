/*!
\file kernels.h
\brief Contains prototypes of functions for array copy, initialization and basic linear algebra operations
*/
#pragma once

#include <cuComplex.h>

typedef cuDoubleComplex DoubleComplex;
class CSR_Matrix;
class Dense_Matrix;


void Copy_array_cpu_to_cpu(const DoubleComplex* cpu_arr_src, DoubleComplex* cpu_arr_dst, const int size);

void Copy_array_gpu_to_gpu(const DoubleComplex* gpu_arr_src, DoubleComplex* gpu_arr_dst, const int size);

int Roundup(const int x, const int y);

bool Is_Finite(const DoubleComplex num);

__global__ void Fill_in_Zeroes_Dense_Matrix(const int rows, const int cols, const int lda, DoubleComplex* mat_val);

__global__ void Fill_in_identity_Dense_Matrix(const int rows, const int lda, DoubleComplex* mat_val); 

void Fill_in_random_numbers_Dense_Matrix(const int rows, const int cols, const int lda, DoubleComplex* mat_values);

//-----------------------------------------------------------------------------------------------------------------

DoubleComplex inner_product_cpu(int N , DoubleComplex* vec1, DoubleComplex* vec2);

double norm_cpu(int N, DoubleComplex* vec);

void axpy_cpu(int N,DoubleComplex* result , DoubleComplex scalar , DoubleComplex* vec);

void scale_vec_cpu(int N,DoubleComplex*vec,DoubleComplex scalar);

void scale_vec_cpu(int N,DoubleComplex*vec, double scalar);

void Gram_Schmidt_cpu(Dense_Matrix& P);

void Gram_Schmidt(Dense_Matrix& P); 

//---------------------------------------------------------------

void Compute_Residual(const CSR_Matrix& A, const Dense_Matrix& b, const Dense_Matrix& x, Dense_Matrix& r);

DoubleComplex Compute_Inner_Product(const Dense_Matrix& vec1, const Dense_Matrix& vec2);

double Compute_L2Norm(const Dense_Matrix& vec);

DoubleComplex Compute_Inner_Product(const Dense_Matrix& mat1, const int col1, const int row1_start, const int row1_end,
    const Dense_Matrix& mat2, const int col2, const int row2_start, const int row2_end);

double Compute_L2Norm(const Dense_Matrix& mat, const int col, const int row_start, const int row_end);

void Compute_HermitianMatrix_vec_mul(const Dense_Matrix& P, const Dense_Matrix& r, Dense_Matrix& f);

void Compute_HermitianMatrix_vec_mul(const Dense_Matrix& matrix, const int col_mat_start, const int col_mat_end, const int row_mat_start, const int row_mat_end,
    const Dense_Matrix& vec, const int col_vec, const int row_vec_start, const int row_vec_end,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end);

void Triangular_Solve(const Dense_Matrix& M, Dense_Matrix& c, const Dense_Matrix& f);

void Triangular_Solve(const Dense_Matrix& mat ,  const int col_start_mat , const int col_end_mat ,const int row_start_mat , const int row_end_mat,
    Dense_Matrix& vec ,const int col_vec , const int row_start_vec , const int row_end_vec , 
   const  Dense_Matrix& rhs, const int col_rhs , const int row_start_rhs , const int row_end_rhs);

void Compute_GeMV(const Dense_Matrix& mat, const int col_mat_start, const int col_mat_end, const int row_mat_start, const int row_mat_end,
    const Dense_Matrix& vec, const int col_vec, const int row_vec_start, const int row_vec_end,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end);

void Compute_Vector_Linear_Combination(const DoubleComplex scalar1, const Dense_Matrix& mat1, const int col_mat1, const int row_mat_start1, const int row_mat_end1,
    const DoubleComplex scalar2, const Dense_Matrix& mat2, const int col_mat2, const int row_mat_start2, const int row_mat_end2,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end);

void Perform_axpy(const DoubleComplex scalar1, const Dense_Matrix& mat1, const int col_mat1, const int row_mat_start1, const int row_mat_end1,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end);

void Compute_CSR_SpMv(const CSR_Matrix& csr_mat,
    const Dense_Matrix& vec, const int col_vec, const int row_vec_start, const int row_vec_end,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end);

void Scaling_Vector(Dense_Matrix& mat, const int col, const int row_start, const int row_end, const  DoubleComplex scalar);               
 
