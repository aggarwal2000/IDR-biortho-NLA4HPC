/*!
\file kernels.cu
\brief Implementation of functions and GPU kernels for array copy, initialization and basic linear algebra operations
*/

#include<iostream>
#include<stdio.h>
#include<cassert>
#include<cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include"Matrix.h"
#include"double_complex.h"
#include"kernels.h"
#include "header.h"
#include "mmio.h"




//---------------------------------------------------------------------------------------------------------------------------

//! Copies elements from source array on CPU to a destination array on CPU
/*!
    \param[in] cpu_arr_src address of the starting location of source array(located on CPU)
    \param[out] cpu_arr_dst address of the starting location of destination array(located on CPU)
    \param[in] size length of the arrays
*/
void Copy_array_cpu_to_cpu(const DoubleComplex* cpu_arr_src, DoubleComplex* cpu_arr_dst, const int size)
{
	for (int i = 0; i < size; i++)
		cpu_arr_dst[i] = cpu_arr_src[i];
}


//! A GPU kernel which copies elements from source array on GPU to a destination array on GPU
/*!
    All elements are copied in parallel.
     One thread is there, per element copy.
   \param[in] src address of starting location of  the source array(present on GPU)
   \param[out] dst base address of starting location of the destination array(present on GPU)
   \param[in] N size of the arrays
*/
__global__ void Copy_array_gpu_to_gpu_kernel(const DoubleComplex* src, DoubleComplex* dst, const int N)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < N)
		dst[id] = src[id];
}


//!  Copies elements from source array on GPU to a destination array on GPU using a GPU kernel
/*!
   \param[in] gpu_arr_src address of the starting location of the source array (on GPU)
   \param[out] gpu_arr_dst address of the staring location of destination array(on GPU)
   \param[in] size length of arrays
*/ 
void Copy_array_gpu_to_gpu(const DoubleComplex* gpu_arr_src, DoubleComplex* gpu_arr_dst, const int size)
{
	dim3 block(256);
	dim3 grid(ceil(static_cast<double>(size) / static_cast<double>(256)));
	Copy_array_gpu_to_gpu_kernel << < grid, block >> > (gpu_arr_src, gpu_arr_dst, size);
}



//! Rounds up a number to a multiple of a given number
/*!
   \param[in] x input number which we want to round up
   \param[in] y given number is to be rounded up to a multiple of y
   \return the required round up
*/
int Roundup(const int x, const int y)
{
    int k = ceil(static_cast<double>(x) / static_cast<double>(y));
    return k * y;
}


//! A GPU kernel which is used to fill in zeroes in  dense matrix object internal GPU values array
/*!
    One thread per matrix row. Rows are handled in parallel.
  \param[in] rows number of rows in dense matrix
  \param[in] cols number of columns in dense matrix
  \param[in] lda leading dimension of dense matrix
  \param[out] mat_val address of the starting location of GPU values array of dense matrix
*/
__global__ void Fill_in_Zeroes_Dense_Matrix(const int rows, const int cols, const int lda, DoubleComplex* mat_val) //Each row handled by one thread
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < rows)
    {

        for (int j = 0; j < cols; j++)
        {
            mat_val[id + j * lda].x = 0;
            mat_val[id + j * lda].y = 0;
        }

    }

}


//! A GPU kernel which is used to initialize a square dense matrix object with Identity matrix
/*!  One thread per matrix row. Rows are handled in parallel
   \param[in] rows number of rows/columns in dense matrix
   \param[in] lds leading dimension of dense matrix
   \param[out] mat_val address of the starting location of GPU values array of dense matrix
*/
__global__ void Fill_in_identity_Dense_Matrix(const int rows, const int lda, DoubleComplex* mat_val) //Each row handled by one thread
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < rows)
    {

        for (int j = 0; j < rows; j++)
        {
            mat_val[id + j * lda].x = 0;
            mat_val[id + j * lda].y = 0;

            if (j == id)
                mat_val[j + j * lda].x = 1;
        }

    }

}


//! Fills in a dense matrix object's internal CPU array with random numbers
/*!
   \param[in] rows number of rows in dense matrix
   \param[in] cols number of columns in dense matrix
   \param[in] lda leading dimension of dense matrix
   \param[out] mat_values address of starting location of CPU values array of dense matrix
*/
void Fill_in_random_numbers_Dense_Matrix(const int rows, const int cols, const int lda, DoubleComplex* mat_values) {

    for (int j = 0; j < cols; j++)
    {
        for (int i = 0; i < rows; i++)
        {
            mat_values[i + j * lda].x = static_cast<double>(rand()) / RAND_MAX;
            mat_values[i + j * lda].y = static_cast<double>(rand()) / RAND_MAX;


        }
    }
}



//-----------------------------------------------------------------------------------------------------------------------------

//! Performs inner product of the vectors(arrays) allocated on CPU memory
/*!
    Performs <vec1,vec2> = (transpose of vec1)*(Conjugate of vec2)
    To get (vec1 hermitian)*(vec2); call for <vec2,vec1>
   \param[in] N size of the arrays
   \param[in] vec1  address of the starting location of first array(allocated on CPU memory)
   \param[in] vec2  address of the starting location of second array(allocated on CPU memory)
   \return <vec1,vec2>
*/
DoubleComplex inner_product_cpu(int N , DoubleComplex* vec1, DoubleComplex* vec2) //<vec1 ,vec2> = vec1 transpose * vec2 conjugate
{
    DoubleComplex temp = { 0,0 };
    for (int i = 0; i < N; i++)
        temp = temp + vec1[i] * conj(vec2[i]);

    return temp;
}



//! Returns L2 norm of vector(array) allocated on CPU memory
/*!
   \param[in] N size of array
   \param[in] vec  address of starting location of the array(allocated on CPU memory)
   \return sqrt(<vec,vec>)
*/
double norm_cpu(int N, DoubleComplex* vec)
{
    DoubleComplex dot = inner_product_cpu(N, vec, vec);
 //   assert(dot.y == 0);
    return sqrt(dot.x);
}




//! Performs axpy operation on vectors(arrays) allocated on CPU
/*!
     y = y + scalar*x, where x and y are arrays allocated on CPU of size N 

     \param[in] N size of array
     \param[in,out] y address of starting location of the array y(allocated on CPU memory)
     \param[in] scalar scalar
     \param[in] x address of starting location of the array(allocated on CPU memory) with which the scalar would be multiplied
*/
void axpy_cpu(int N,DoubleComplex* y , DoubleComplex scalar , DoubleComplex* x)
{
    for(int i=0;i<N;i++)
        y[i] = y[i] + scalar*x[i];
}

//! Scales a vector(array allocated on CPU)
/*!
   vec = scalar*vec , where vec is an array on CPu of size N
*/
void scale_vec_cpu(int N,DoubleComplex*vec,DoubleComplex scalar)
{
  for(int i=0;i<N;i++)
    vec[i] = scalar*vec[i];
}


//! Scales a vector(array allocated on CPU)
/*!
   vec = scalar*vec , where vec is an array on CPU of size N
*/
void scale_vec_cpu(int N,DoubleComplex*vec, double scalar)
{
  for(int i=0;i<N;i++)
    vec[i] = scalar*vec[i];
}


//! Performs Gram Schmidt orthogonalization process on columns(matrix values allocated on CPU) of a dense matrix
/*!
\param[in,out] Dense matrix whose columns are to be orthonormalized
*/
void Gram_Schmidt_cpu(Dense_Matrix& P)  
{  
    assert(P.ExistsCPU() == true);
    DoubleComplex* u_i = new DoubleComplex[P.GetRows()];
   
    for (int i = 0; i < P.GetCols(); i++)
    {
        //ui = vi
        DoubleComplex* v_i = P.GetCPUValues() + P.GetLda() * i; //i is col num
        Copy_array_cpu_to_cpu(v_i, u_i, P.GetRows());

        for (int j = 0; j < i; j++)
        {
            //ui = ui - Proj(vi on uj)
           
            DoubleComplex* u_j = P.GetCPUValues() + P.GetLda() * j; //can also use GetColPtrfn(j) of dense matrix class here 
          
            DoubleComplex mul = inner_product_cpu(P.GetRows(), v_i, u_j) / inner_product_cpu(P.GetRows() , u_j , u_j);

            axpy_cpu(P.GetRows() ,u_i , -1*mul, u_j);

        }

        //vi = ui
        Copy_array_cpu_to_cpu(u_i, v_i, P.GetRows());
    }

    delete[] u_i;

    for (int i = 0; i < P.GetCols(); i++)
    {
        //ui = ui/||ui|| for all cols of P
        DoubleComplex* vec = P.GetCPUValues() + P.GetLda() * i;
        double nrm = norm_cpu(P.GetRows() , vec);
       
        scale_vec_cpu(P.GetRows(),vec,1/nrm);
    }
   
}




//! Returns true if the complex number is finite(that is Re(z) and Im(z) are not nan or inf)
bool Is_Finite(const DoubleComplex num)
{
    if (std::isfinite(num.x) && std::isfinite(num.y))
        return true;
    else
        return false;
}


//-----------------------------------------------------------------------------------------------------------------------------


//! A GPU kernel which computes residual vector 
/*!
    Computes r = b - A*x (A is a sparse CSR matrix)
    Each element of the residual vector is computed using one warp. All elements are computed in parallel.
    It uses warp __shfl_down_sync.
    \param[in] N length of vectors/arrays
    \param[in] row_ptr_A base address of the row pointers array of CSR matrix A (allocated on GPU memory)
    \param[in] col_ind_A base address of the column indices array of CSR matrix A (allocated on GPU memory)
    \param[in] val_A base address of the values array of CSR matrix A (allocated on GPU memory)
    \param[in] b  address of starting location of the array storing vector b(allocated on GPU memory) 
    \param[in] x  address of starting location of the array storing vector x(alloacted on GPU memory)
    \param[out] r  address of starting location of the array where residual is to be stored (present on GPU memory)
*/
__global__ void Compute_residual_kernel(const int N, const int* row_ptr_A, const int* col_ind_A,
 const DoubleComplex* val_A, const  DoubleComplex* b, const DoubleComplex* x, DoubleComplex* r)
{
    //per row , we've 1 warp.
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

    int row = warp_index;

    if (row < N) {
        int start_index_of_row = row_ptr_A[row]; //inclusive
        int end_index_of_row = row_ptr_A[row + 1]; //exclusive

        int id_within_warp = gid % 32;


        DoubleComplex temp = { 0,0 };

        for (int k = start_index_of_row + id_within_warp; k < end_index_of_row; k = k + 32)
        {
            temp = temp + val_A[k] * x[col_ind_A[k]];
        }



        DoubleComplex val = temp;

       // double re;
       // double im;
        for (int offset = 16; offset > 0; offset /= 2)
        {
            //re = val.x;
            //im = val.y;
            //re += __shfl_down_sync(FULL_MASK, re, offset);
            //im += __shfl_down_sync(FULL_MASK, im, offset);
            val.x += __shfl_down_sync(FULL_MASK, val.x, offset);
            val.y += __shfl_down_sync(FULL_MASK, val.y, offset);
        }

        if (id_within_warp == 0)
            //r[row] = b[row] - make_cuDoubleComplex(re, im);
            r[row] = b[row] - val;

    }

}


//! Computes residual vector using a GPU kernel
/*!
   performs  r = b - A*x
   \param[in] A CSR Matrix object reference
   \param[in] b RHS b(dense matrix object reference)
   \param[in] x Solution approx. x (dense matrix object reference)
   \param[out] r residual r (dense matrix object reference)

  Operates on/uses internal GPU arrays of the matrices/vectors.
*/
void Compute_Residual(const CSR_Matrix& A, const Dense_Matrix& b, const Dense_Matrix& x, Dense_Matrix& r)
{   
    assert(A.ExistsGPU() == true);
    assert(b.ExistsGPU() == true);
    assert(x.ExistsGPU() == true);
    assert(r.ExistsGPU() == true);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(static_cast<double>(A.GetRows()) * WARP_SIZE / static_cast<double>(THREADS_PER_BLOCK)));
    Compute_residual_kernel << < grid, block >> > (A.GetRows(), A.GetGPURowPtr(), A.GetGPUColInd(), A.GetGPUValues(), b.GetGPUValues(), x.GetGPUValues(), r.GetGPUValues());
}





//----------------------------------------------------------------------------------------------------------------------------



//! Performs parallel reduction on data array using a thread block
/*!
     \param[in,out] data base address of GPU array  
*/
__device__ void block_reduce(DoubleComplex* data)
{
    int nt = blockDim.x;
    int tid = threadIdx.x;

    for (int k = nt / 2; k > 0; k = k / 2)
    {
        __syncthreads();
        if (tid < k)
        {
            data[tid] += data[tid + k];
        }
    }
}



//! A helper GPU kernel which contributes to computing of inner product of vectors
/*!
   \param[in] N size of vector/array
   \param[in] vec1 address of starting location of first (vector)array(allocated on GPU) 
   \param[in] vec2 address of starting location of second (vector)array (allocated on GPU)
   \param[out] buffer base address of buffer/workspace(allocated on GPU memory)

*/
__global__ void inner_product_kernel1(const int N, const DoubleComplex* vec1, const DoubleComplex* vec2,
     DoubleComplex* buffer) {

    int dimgrid = gridDim.x;
    int dimblock = blockDim.x;

    int lid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    DoubleComplex tmp = { 0,0 };
    for (int i = gid; i < N; i += dimgrid * dimblock)
    {
        tmp += vec1[i] * conj(vec2[i]);
    }

    __shared__ DoubleComplex tmp_work[THREADS_PER_BLOCK];
    tmp_work[lid] = tmp;

    __syncthreads();
    block_reduce(tmp_work);

    if (lid == 0)
        buffer[blockIdx.x] = tmp_work[0];
}



//! A helper GPU kernel which contributes to computing of inner product of vectors
/*!
   \param[in] arr_size  size of buffer array
   \param[in] buffer base address of buffer array(allocated on GPU)
   \param[out] ans pointer to gpu memory location where the answer of the inner product is to be stored

*/
__global__ void inner_product_kernel2(const int arr_size, DoubleComplex* buffer, DoubleComplex* ans)
{
    int nt = blockDim.x;
    int tid = threadIdx.x;

    DoubleComplex temp = { 0,0 };
    for (int i = tid; i < arr_size; i += nt)
    {
        temp += buffer[i];
    }

    __shared__ DoubleComplex tmp_work[THREADS_PER_BLOCK];
    tmp_work[tid] = temp;

    __syncthreads();
    block_reduce(tmp_work);

    if (tid == 0)
        *ans = tmp_work[0];
}


//! Computes inner product of 2 vectors using GPU kernels
/*! 
    <vec1,vec2> = (vec1 transposed)*(vec2 conjugate)
    
     To get (vec1 hermitian)*(vec2); call function for <vec2,vec1> 
    \param[in] vec1 first vector(reference to a dense matrix object)
    \param[in] vec2 second vector(reference to a dense matrix object)
    \return <vec1,vec2>

    Operates on internal GPU arrays of the vectors.
*/
DoubleComplex Compute_Inner_Product(const Dense_Matrix& vec1, const Dense_Matrix& vec2) { //does <u,v> = u transpose * v conjugate
   //return dot product of 2 vectors...                                              // <v,u> = u hermitian * v
   
    assert(vec1.ExistsGPU() == true);
    assert(vec2.ExistsGPU()== true);

    DoubleComplex ans;
    DoubleComplex* gpu_ans;
    cudaMalloc((void**)&gpu_ans, sizeof(DoubleComplex));

    dim3 block(THREADS_PER_BLOCK);
    int work_per_thread = 4;
    int N = vec1.GetRows();
    // int gridsize = ceil(static_cast<double>(N) / static_cast<double>(THREADS_PER_BLOCK * work_per_thread));
    int gridsize = ceil((double)N / (double)(THREADS_PER_BLOCK * work_per_thread));
    dim3 grid(gridsize);
    DoubleComplex* gpu_buffer;
    const int buffer_size = (gridsize + 1);
    cudaMalloc((void**)&gpu_buffer, buffer_size * sizeof(DoubleComplex));


    inner_product_kernel1 << < grid, block >> > (N, vec1.GetGPUValues(), vec2.GetGPUValues(), gpu_buffer);
    inner_product_kernel2 << < 1, block >> > (gridsize, gpu_buffer, gpu_ans);

    cudaMemcpy(&ans, gpu_ans, sizeof(DoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(gpu_buffer);
    cudaFree(gpu_ans);
    return ans;
}



//! Computes L2 norm of the vector using GPU kernels
/*! 
     ||vec|| = sqrt(<vec,vec>)
   \param[in] vec vector(refernece to dense matrix object)
   \return ||vec||
  
   Uses or operates on internal GPU array of the vector
*/
double Compute_L2Norm(const Dense_Matrix& vec)
{
    assert(vec.ExistsGPU() == true);
    DoubleComplex inner_product = Compute_Inner_Product(vec, vec); //Inner product will return a real number
    //assert(imag(inner_product) == 0);
    return sqrt(real(inner_product));

}


//! Computes inner product of 2 vectors using GPU kernels
/*!
    Performs <vec1,vec2> = (vec1 transposed)*(vec2 conjugate)
    <br> To get (hermitian transpose of vec1)*(vec2); call this function to compute <vec2,vec1>
    
    The vector is defined using a dense matrix by starting and ending rows and a column
  
   \param[in] mat1 dense matrix1
   \param[in] col1 column of matrix1 specifying the vector
   \param[in] row1_start  row of matrix1 specifying the start of vector
   \param[in] row1_end  row of matrix1 specifying the end of vector
   \param[in] mat2 dense matrix2
   \param[in] col2 column of matrix2 specifying the vector
   \param[in] row2_start  row of matrix2 specifying the start of vector
   \param[in] row2_end row of matrix2 specifying the end of vector
   \return <vec1,ve2>

    This function uses/operates on internal GPU arrays of vectors/matrix.
*/
DoubleComplex Compute_Inner_Product(const Dense_Matrix& mat1, const int col1, const int row1_start, const int row1_end,
    const Dense_Matrix& mat2, const int col2, const int row2_start, const int row2_end) { //does <u,v> = u transpose * v conjugate
   //return dot product of 2 vectors...                                              // <v,u> = u hermitian * v
    
    assert(mat1.ExistsGPU() == true);
    assert(mat2.ExistsGPU() == true);

    DoubleComplex ans;
    DoubleComplex* gpu_ans;
    cudaMalloc((void**)&gpu_ans, sizeof(DoubleComplex));

    dim3 block(THREADS_PER_BLOCK);
    int work_per_thread = 4;
    int N = row2_end - row2_start + 1;
    assert((row2_end - row2_start) == (row1_end - row1_start));
    assert(col1 < mat1.GetCols());
    assert(col2 < mat2.GetCols());
    assert(row1_end < mat1.GetRows());
    assert(row2_end < mat2.GetRows());
    int gridsize = ceil(static_cast<double>(N) / static_cast<double>(THREADS_PER_BLOCK * work_per_thread));
    dim3 grid(gridsize);
    DoubleComplex* gpu_buffer;
    const int buffer_size = (gridsize + 1);
    cudaMalloc((void**)&gpu_buffer, buffer_size * sizeof(DoubleComplex));

    DoubleComplex* gpu_vec1 = mat1.GetGPUValues() + row1_start + mat1.GetLda() * col1;
    DoubleComplex* gpu_vec2 = mat2.GetGPUValues() + row2_start + mat2.GetLda() * col2;
    inner_product_kernel1 << < grid, block >> > (N, gpu_vec1, gpu_vec2, gpu_buffer);
    inner_product_kernel2 << < 1, block >> > (gridsize, gpu_buffer, gpu_ans);

    cudaMemcpy(&ans, gpu_ans, sizeof(DoubleComplex), cudaMemcpyDeviceToHost);
    cudaFree(gpu_buffer);
    cudaFree(gpu_ans);
    return ans;
}



//! Computes L2 Norm of a  vector using GPU kernels
/*!
    Performs ||vec|| = sqrt(<vec,vec>)
    
    The vector is defined using a dense matrix by starting and ending rows and a column
  
   \param[in] mat dense matrix
   \param[in] col column of matrix specifying the vector
   \param[in] row_start  row of matrix specifying the start of vector
   \param[in] row_end  row of matrix specifying the end of vector
   \return ||vector||

    This function uses/operates on internal GPU arrays of vectors/matrix.
*/
double Compute_L2Norm(const Dense_Matrix& mat, const int col, const int row_start, const int row_end)
{   
    assert(mat.ExistsGPU() == true);
    DoubleComplex inner_product = Compute_Inner_Product(mat, col, row_start, row_end, mat, col, row_start, row_end); //This Inner product will return a real number
   // assert(imag(inner_product) == 0);
  // std::cout << "\n\n imag(inner_product):" << imag(inner_product) << std::endl;
    return sqrt(real(inner_product));

}




//--------------------------------------------------------------------------------------------------------------------




//! GPU kernel which performs mutiplication of conjugate transpose of a dense matrix with a vector
/*! 
    result = (hermitian transpose of matrix)*(vector)
    Per element of the result vector, one warp is used. All elements of the result are computed in parallel. 
    \param[in] mat_rows number of rows in the dense matrix
    \param[in] mat_cols number of columns in the dense matrix
    \param[in] lda_mat leading dimension of the dense matrix
    \param[in] mat_values  address of the starting location of GPU values array of the matrix.
    \param[in] vec2 address of the starting location of GPU array storing vector to be multiplied
    \param[out] vec_result address of the starting location of GPU array where the result is to be stored
*/
__global__ void HermitianMatrix_vec_mul(const int mat_rows, const int mat_cols, const int lda_mat,
     const DoubleComplex* mat_values, const DoubleComplex* vec2, DoubleComplex* vec_result)
{
    
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_index = gid/WARP_SIZE;
    int id_within_warp = gid%WARP_SIZE;

    if(warp_index < mat_cols)
    {
      DoubleComplex* vec1 = const_cast<DoubleComplex*>(mat_values) + warp_index * lda_mat;

      DoubleComplex temp = {0,0};

      for(int k= id_within_warp ; k < mat_rows ; k = k+32)
      {
        temp = temp + conj(vec1[k])*vec2[k];
      }

      DoubleComplex val = temp;

      for (int offset = 16; offset > 0; offset /= 2)
      {
          val.x += __shfl_down_sync(FULL_MASK, val.x, offset);
          val.y += __shfl_down_sync(FULL_MASK, val.y, offset);
      }

      if(id_within_warp == 0)
        vec_result[warp_index] = val;

    }

}


//! Performs mutiplication of conjugate transpose of a dense matrix with a vector using GPU kernel
/*!
    Does: f = (hermitian transpose of P)*r
    \param[in] P dense matrix(reference to a dense matrix object) 
    \param[in] r vector (reference to a dense matrix object)
    \param[out] f result (reference to a dense matrix object)

    It uses/operates on internal GPU arrays of matrix and vector.
*/
void Compute_HermitianMatrix_vec_mul(const Dense_Matrix& P, const Dense_Matrix& r, Dense_Matrix& f)
{   
    assert(P.ExistsGPU() == true);
    assert(r.ExistsGPU() == true);
    assert(f.ExistsGPU() == true);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(static_cast<double>(P.GetCols() * WARP_SIZE) / static_cast<double>(THREADS_PER_BLOCK)));
  HermitianMatrix_vec_mul << < grid, block >> > (P.GetRows(), P.GetCols(), P.GetLda(), P.GetGPUValues(), r.GetGPUValues(), f.GetGPUValues());
}


//! Performs multiplication of conjugate transpose of a matrix with a vector using GPU kernel
/*!
    result = (conjugate transpose of a submatrix)*(vector)

    Vector is defined using a dense matrix by starting and ending rows and a column
     
    Matrix is the submatrix of a dense matrix object defined by starting and ending rows and columns.
  
    This function uses/operates on internal GPU arrays of vectors/matrix.

*/
void Compute_HermitianMatrix_vec_mul(const Dense_Matrix& matrix, const int col_mat_start, const int col_mat_end, const int row_mat_start, const int row_mat_end,
    const Dense_Matrix& vec, const int col_vec, const int row_vec_start, const int row_vec_end,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end)
{  
    
    assert(matrix.ExistsGPU() == true);
    assert(vec.ExistsGPU() == true);
    assert(result.ExistsGPU() == true);
    assert((row_mat_end - row_mat_start) == (row_vec_end - row_vec_start));
    assert((col_mat_end - col_mat_start) == (row_result_end - row_result_start));
    int rows_mat = row_mat_end - row_mat_start + 1; //rows in matrix
    int cols_mat = col_mat_end - col_mat_start + 1;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(static_cast<double>(cols_mat*WARP_SIZE) / static_cast<double>(THREADS_PER_BLOCK)));
    int lda_mat = matrix.GetLda();
    DoubleComplex* gpu_mat_values = matrix.GetGPUValues() + col_mat_start * matrix.GetLda() + row_mat_start;
    DoubleComplex* gpu_vec_values = vec.GetGPUValues() + col_vec * vec.GetLda() + row_vec_start;
    DoubleComplex* gpu_result_values = result.GetGPUValues() + col_result * result.GetLda() + row_result_start;
    HermitianMatrix_vec_mul << < grid, block >> > (rows_mat, cols_mat, lda_mat, gpu_mat_values, gpu_vec_values, gpu_result_values);
}



//-----------------------------------------------------------------------------------------------------------------------



//! Solves lower triangular system
/*!
    solves c from Mc = f
    \param[in] M Lower triangular matrix (reference to a dense matrix object) 
    \param[in] c vector to be computed (reference to a dense matrix object) 
    \param[out] f rhs (reference to a dense matrix object) 

    Uses/Operates on internal CPU arrays of the matrix and vectors 
*/ 
void Triangular_Solve(const Dense_Matrix& M, Dense_Matrix& c, const Dense_Matrix& f)
{
   
    assert(M.ExistsCPU() == true);
    assert(c.ExistsCPU() == true);
    assert(f.ExistsCPU() == true);
    for (int row = 0; row < M.GetRows(); row++)
    {
        DoubleComplex sum = { 0,0 };
        for (int col = 0; col < row; col++)
        {
            sum = sum + (M.GetCPUValues())[row + col * M.GetLda()] * (c.GetCPUValues())[col];
            
        }
       
        c.GetCPUValues()[row] = (f.GetCPUValues()[row] - sum) / (M.GetCPUValues())[row + row * M.GetLda()];
    }

  
}


//! Solves a lower triangular system
/*!
    solves vec from mat*vec = rhs
    
    Vector is defined using a dense matrix by starting and ending rows and a column
     
    Matrix is the submatrix of a dense matrix object defined by starting and ending rows and columns.
  
    This function uses/operates on internal CPU arrays of vectors/matrix.
*/
void Triangular_Solve(const Dense_Matrix& mat ,  const int col_start_mat , const int col_end_mat ,const int row_start_mat , const int row_end_mat,
    Dense_Matrix& vec ,const int col_vec , const int row_start_vec , const int row_end_vec , 
   const  Dense_Matrix& rhs, const int col_rhs , const int row_start_rhs , const int row_end_rhs)
{
    assert(mat.ExistsCPU() == true);
    assert(vec.ExistsCPU() == true);
    assert(rhs.ExistsCPU() == true);
    assert((col_end_mat - col_start_mat) == (row_end_vec - row_start_vec));
    assert((row_end_mat - row_start_mat) == (row_end_rhs - row_start_rhs));

   // DoubleComplex* matrix = mat.GetCPUValues() + row_start_mat + mat.GetLda() * col_start_mat;
    DoubleComplex* matrix = mat.GetSpecificLocationPtrCPU(row_start_mat, col_start_mat); //or can directly work on submatrix instead of this by altering start and end of for loop
    DoubleComplex* vector = vec.GetSpecificLocationPtrCPU(row_start_vec, col_vec);
    DoubleComplex* right = rhs.GetSpecificLocationPtrCPU(row_start_rhs, col_rhs);
    int mat_rows = row_end_mat - row_start_mat + 1;
    //int mat_cols = col_end_mat - col_start_mat + 1;

    for (int row = 0; row < mat_rows ; row++)
    {
        DoubleComplex sum = { 0,0 };
        for (int col = 0; col < row; col++)
        {
            sum = sum +   *(matrix + row + col * mat.GetLda()) * (vector)[col];

        }

        vector[row] = (right[row] - sum) / *(matrix + row + row * mat.GetLda());
       
    }
    
}



//-------------------------------------------------------------------------------------------------------------------




//! GPU kernel which performs dense matrix vector mutiplication
/*!  
    result = dense matrix*vec
    Each row of the dense matrix is handled using one thread. All elements of the result vector are computed in parallel. 
   \param[in] mat_rows number of rows in dense matrix
   \param[in] mat_cols number of columns in dense matrix
   \param[in] mat_lda leading dimension of matrix
   \param[in] mat_values address of starting location of GPU values array of matrix
   \param[in] vec address of starting location of GPU values array of the vector to mutiplied
   \param[out] result address of starting location of GPU values array where the result is to be stored
*/ 
__global__ void GeMV(const int mat_rows, const int mat_cols, const int mat_lda, 
    const DoubleComplex* mat_values, const DoubleComplex* vec, DoubleComplex* result)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < mat_rows)
    {
        DoubleComplex sum = { 0,0 };
        for (int col = 0; col < mat_cols; col++)
            sum += mat_values[row + col * mat_lda] * vec[col];

        result[row] = sum;
    }
}

//! Performs general matrix vector mutiplication using GPU kernel
/*!
     result = matrix*vec
    
    Vector is defined using a dense matrix by starting and ending rows and a column
     
    Matrix is the submatrix of a dense matrix object defined by starting and ending rows and columns.
  
    This function uses/operates on internal GPU arrays of vectors/matrix.
*/
void Compute_GeMV(const Dense_Matrix& mat, const int col_mat_start, const int col_mat_end, const int row_mat_start, const int row_mat_end,
    const Dense_Matrix& vec, const int col_vec, const int row_vec_start, const int row_vec_end,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end)
{
    assert(mat.ExistsGPU() == true);
    assert(vec.ExistsGPU() == true);
    assert(result.ExistsGPU() == true);
    assert((row_mat_end - row_mat_start) == (row_result_end - row_result_start));
    assert((col_mat_end - col_mat_start) == (row_vec_end - row_vec_start));
    int rows_mat = row_mat_end - row_mat_start + 1; //rows in matrix
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(static_cast<double>(rows_mat) / static_cast<double>(THREADS_PER_BLOCK)));
    int cols_mat = col_mat_end - col_mat_start + 1;
    int lda_mat = mat.GetLda();
    DoubleComplex* gpu_mat_values = mat.GetGPUValues() + col_mat_start * mat.GetLda() + row_mat_start;
    DoubleComplex* gpu_vec_values = vec.GetGPUValues() + col_vec * vec.GetLda() + row_vec_start;
    DoubleComplex* gpu_result_values = result.GetGPUValues() + col_result * result.GetLda() + row_result_start;
    GeMV << < grid, block >> > (rows_mat, cols_mat, lda_mat, gpu_mat_values, gpu_vec_values, gpu_result_values);

}




//--------------------------------------------------------------------------------------------------------------------






//! GPU kernel which computes linear combination of vectors
/*!  
     vec_result = scalar1*vec1 + scalar2*vec2

     If any vector participating in the linear combination is null, it is left out.
     In case, all are null, then the result is initialized with zeroes. 

     One thread per element of the result. All elements of the result are computed in parallel.
    
     \param[in] N length of arrays
     \param[out] vec_result address of starting location of array on GPU where the result is to be stored
     \param[in] scalar1 scalar1
     \param[in] vec1 address of starting location of array(on GPU) corresponding to first vector
     \param[in] scalar2 scalar2
     \param[in] vec2 address of starting location of array(on GPU) corresponding to second vector

*/
__global__ void Vector_Linear_Combination(const int N, DoubleComplex* vec_result, const DoubleComplex scalar1, const  DoubleComplex* vec1, const DoubleComplex scalar2, const  DoubleComplex* vec2)
{

    int row = blockDim.x * blockIdx.x + threadIdx.x; //global thread id

    if (row < N)
    {
        if (vec1 == nullptr && vec2 == nullptr)
        {
            vec_result[row] = { 0,0 };
        }
        else if (vec1 == nullptr)
            vec_result[row] = scalar2 * vec2[row];
        else if (vec2 == nullptr)
        {
            vec_result[row] = scalar1 * vec1[row];
        }
        else
            vec_result[row] = scalar1 * vec1[row] + scalar2 * vec2[row];
    }
}



//! Computes linear combination of vectors using GPU kernel
/*!
     result = scalar1*vec1 + scalar2*vec2
    
    Vector is defined using a dense matrix by starting and ending rows and a column.
  
    This function uses/operates on internal GPU arrays of vectors/matrix.
*/
void Compute_Vector_Linear_Combination(const DoubleComplex scalar1, const Dense_Matrix& mat1, const int col_mat1, const int row_mat_start1, const int row_mat_end1,
    const DoubleComplex scalar2, const Dense_Matrix& mat2, const int col_mat2, const int row_mat_start2, const int row_mat_end2,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end)
{  

    int N = row_result_end - row_result_start + 1;

    DoubleComplex* gpu_vec1 = nullptr;
    DoubleComplex* gpu_vec2 = nullptr;
    if (scalar1.x == 0.0 && scalar1.y == 0.0)
        gpu_vec1 = nullptr;
    else
    {
        assert((row_mat_end1 - row_mat_start1 + 1) == N);
        gpu_vec1 = mat1.GetGPUValues() + col_mat1 * mat1.GetLda() + row_mat_start1;
    }

    if (scalar2.x == 0.0 && scalar2.y == 0.0)
        gpu_vec2 = nullptr;
    else
    {
        assert((row_mat_end2 - row_mat_start2 + 1) == N);
        gpu_vec2 = mat2.GetGPUValues() + col_mat2 * mat2.GetLda() + row_mat_start2;
    }

    /* if (!(scalar1.x == 0.0 && scalar1.y == 0.0))
     {
         assert((row_mat_end1 - row_mat_start1 + 1) == N);
         gpu_vec1 = mat1.GetGPUValues() + col_mat1 * mat1.GetLda() + row_mat_start1;
     }
     if (!(scalar2.x != 0.0 && scalar2.y != 0.0))
     {
         assert((row_mat_end2 - row_mat_start2 + 1) == N);
         gpu_vec2 = mat2.GetGPUValues() + col_mat2 * mat2.GetLda() + row_mat_start2;
     } */
    DoubleComplex* gpu_result = result.GetGPUValues() + col_result * result.GetLda() + row_result_start;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(static_cast<double>(N) / static_cast<double>(THREADS_PER_BLOCK)));
    Vector_Linear_Combination << < grid, block >> > (N, gpu_result, scalar1, gpu_vec1, scalar2, gpu_vec2);
}



//! GPU kernel which performs axpy operation
/*!  
     vec_result = scalar1*vec1 + vec_result


     One thread per element of the result. All elements of the result are computed in parallel.
    
     \param[in] N length of arrays
     \param[in,out] vec_result address of starting location of array on GPU where the result is to be stored
     \param[in] scalar scalar
     \param[in] vec1 address of starting location of array(on GPU) corresponding to first vector
     

*/
__global__ void axpy_kernel(const int N, DoubleComplex* vec_result, const DoubleComplex scalar, const  DoubleComplex* vec1)
{

    int row = blockDim.x * blockIdx.x + threadIdx.x; //global thread id

    if (row < N)
    {
            vec_result[row] = scalar * vec1[row] +  vec_result[row];
    }
}



//! Performs axpy operation using GPU kernel
/*!
     result = scalar*vec1 + result
    
    Vector is defined using a dense matrix by starting and ending rows and a column.
  
    This function uses/operates on internal GPU arrays of vectors/matrix.
*/
void Perform_axpy(const DoubleComplex scalar1, const Dense_Matrix& mat1, const int col_mat1, const int row_mat_start1, const int row_mat_end1,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end)
{  

    int N = row_result_end - row_result_start + 1;
    assert(mat1.ExistsGPU() == true);
    assert(result.ExistsGPU() == true);
    DoubleComplex* gpu_vec1 = nullptr;
    assert((row_mat_end1 - row_mat_start1 + 1) == N);
    gpu_vec1 = mat1.GetGPUValues() + col_mat1 * mat1.GetLda() + row_mat_start1;
    DoubleComplex* gpu_result = result.GetGPUValues() + col_result * result.GetLda() + row_result_start;
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(static_cast<double>(N) / static_cast<double>(THREADS_PER_BLOCK)));
    axpy_kernel<< < grid, block >> > (N, gpu_result, scalar1, gpu_vec1);
}




//-----------------------------------------------------------------------------------------------------------------------

//! GPU kernel which performs CSR sparse matrix vector mutiplication
/*!
     vec_result = CSR matrix*vec
     Each row of sparse matrix is handled using a warp. Uses __shfl_down_sync.
     \param[in] N length of arrays/vectors
     \param[in] row_ptr_matrix address of starting location of row pointers array(present on GPU) of CSR matrix
     \param[in] col_ind_matrix address of starting location of coulm indices array(present on GPU) of CSR matrix
     \param[in] val_matrix  address of starting location of values array(present on GPU) of CSR matrix
     \param[in] vec address of starting location of the GPU array corresonding to vector to be multiplied
     \param[out] vec_result address of starting location of the GPU array where the result is to be stored.
*/
__global__ void CSR_SpMV(int N, const int* row_ptr_matrix, const int* col_ind_matrix, const DoubleComplex* val_matrix, const DoubleComplex* vec, DoubleComplex* vec_result)
{
    //per row , we've 1 warp.
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int warp_index = (int)(gid / WARP_SIZE);  //(32 is warp size)

    int row = warp_index;

    if (row < N) {
        int start_index_of_row = row_ptr_matrix[row]; //inclusive
        int end_index_of_row = row_ptr_matrix[row + 1]; //exclusive

        int id_within_warp = gid % 32;


        DoubleComplex temp = { 0,0 };

        for (int k = start_index_of_row + id_within_warp; k < end_index_of_row; k = k + 32)
        {
            temp = temp + val_matrix[k] * vec[col_ind_matrix[k]];
        }

        DoubleComplex val = temp;

        for (int offset = 16; offset > 0; offset /= 2)
        {
            val.x += __shfl_down_sync(FULL_MASK, val.x, offset);
            val.y += __shfl_down_sync(FULL_MASK, val.y, offset);
        }

        if (id_within_warp == 0)
            vec_result[row] = val;

    }

}




//! Performs CSR matrix vector mutiplication using GPU kernel
/*!
     result = CSR matrix*vec
    
    Vector is defined using a dense matrix by starting and ending rows and a column
  
    This function uses/operates on internal GPU arrays of vectors/matrix.
*/
void Compute_CSR_SpMv(const CSR_Matrix& csr_mat,
    const Dense_Matrix& vec, const int col_vec, const int row_vec_start, const int row_vec_end,
    Dense_Matrix& result, const int col_result, const int row_result_start, const int row_result_end)
{ //add more assert statements for matrices
    
    assert(csr_mat.ExistsGPU() == true);
    assert(vec.ExistsGPU() == true);
    assert(result.ExistsGPU() == true);
    int rows_mat = csr_mat.GetRows();
    int cols_mat = csr_mat.GetCols();
    assert(cols_mat == (row_vec_end - row_vec_start + 1));
    assert(rows_mat == (row_result_end - row_result_start + 1));
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(static_cast<double>(rows_mat * WARP_SIZE) / static_cast<double>(THREADS_PER_BLOCK)));
    DoubleComplex* gpu_vec_values = vec.GetGPUValues() + col_vec * vec.GetLda() + row_vec_start;
    DoubleComplex* gpu_result_values = result.GetGPUValues() + col_result * result.GetLda() + row_result_start;

    CSR_SpMV << < grid, block >> > (rows_mat, csr_mat.GetGPURowPtr(), csr_mat.GetGPUColInd(), csr_mat.GetGPUValues(), gpu_vec_values, gpu_result_values);
}



//---------------------------------------------------------------------------------------------------------------


//! GPU kernel which scales a vector
/*!
    One thread, per component of the vector
    \param[in] N size of array
    \param[in,out] vec address of starting location of GPU array corresponding to vector
    \param scalar scalar  
*/
__global__ void scaling_vec(const int N, DoubleComplex* vec, DoubleComplex scalar)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N)
    {
        vec[id] = scalar * vec[id];
    }
}



//! Scales a vector using GPU kernel
/*!
     vec = scalar*vec
    
    Vector is defined using a dense matrix by starting and ending rows and a column.
  
    This function uses/operates on internal GPU arrays of vectors/matrix.
*/
void Scaling_Vector(Dense_Matrix& mat, const int col, const int row_start, const int row_end, const  DoubleComplex scalar)
{  
    assert(mat.ExistsGPU() == true);
    DoubleComplex* gpu_vec = mat.GetGPUValues() + row_start + col * mat.GetLda();
    int N = row_end - row_start + 1;
    assert(N >= 0);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(static_cast<double>(N) / static_cast<double>(THREADS_PER_BLOCK)));
    scaling_vec << <grid, block >> > (N, gpu_vec, scalar);
}


//------------------------------------------------------------------------------------------------------------------------



//! Performs Gram Schmidt orthogonalization process on columns(matrix values allocated on GPU) of a dense matrix
/*!
   \param[in,out] Dense matrix whose columns are to be orthonormalized
*/
void Gram_Schmidt(Dense_Matrix& P)  
{  
    assert(P.ExistsGPU() == true);

    Dense_Matrix u_i(P.GetRows(), 1, P.GetRows(), ORDER::COLUMN_MAJOR, CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::EXISTENT);
   
    for (int i = 0; i < P.GetCols(); i++)
    {
        //ui = vi
        DoubleComplex* v_i = P.GetColPtrGPU(i); //i is col num
        Copy_array_gpu_to_gpu(v_i, u_i.GetGPUValues(), P.GetRows());
        

        for (int j = 0; j < i; j++)
        {
            //ui = ui - Proj(vi on uj)
           
           // mul = inner_product_cpu(P.GetRows(), v_i, u_j) / inner_product_cpu(P.GetRows() , u_j , u_j);
            DoubleComplex mul = Compute_Inner_Product(P,i,0,P.GetRows()-1,P,j,0,P.GetRows()-1)/Compute_Inner_Product(P,j,0,P.GetRows()-1,P,j,0,P.GetRows()-1);
            
           //u_i = u_i - mul*u_j; 
            Perform_axpy(-1*mul , P,j,0,P.GetRows()-1, u_i,0,0,P.GetRows()-1);

        }

        //vi = ui
        Copy_array_gpu_to_gpu(u_i.GetGPUValues(), v_i, P.GetRows());
    }

  

    for (int i = 0; i < P.GetCols(); i++)
    {
        //ui = ui/||ui|| for all cols of P
       double nrm = Compute_L2Norm(P,i,0,P.GetRows()-1);
       Scaling_Vector(P,i,0,P.GetRows()-1,{1/nrm,0});
        
    }
   
}
