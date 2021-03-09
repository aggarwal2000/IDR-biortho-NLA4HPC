/*!
\file Preconditioner.cu
 \brief Implementation of member functions of classes JacobiPreconditioner, RichardsonPreconditioner and other functions and kernels related to preconditioners 
*/

#include <stdio.h>
#include <iostream>
#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "double_complex.h"
#include "Matrix.h"
#include "Preconditioner.h"
#include "kernels.h"
#include "header.h"


//------------------------------------------------------------------------------------------------------------------------
                                /* Functions amd kernels for JacobiPreconditioner class */



//! GPU kernel which fills in the Jacobi Preconditioner values 
/*!
  Per element of the preconditioner's internal array, one thread is used. All elements are filled in, in parallel. 
 \param[in] N dimension of the square sparse matrix
 \param[in] row_ptr_matrix base address of array allocated on GPU which stores CSR matrix row pointers 
 \param[in] col_ind_matrix base address of array allocated on GPU which stores CSR matrix column indices
 \param[in] values_matrix  base address of array allocated on GPU which stores CSR matrix values
 \param[out] d_inverse base address of array allocated on GPU which is to be filled with Jacobi preconditioner values
*/ 
__global__ void Diagonal_Scaling_Jacobi(const int N, const int* row_ptr_matrix, const int* col_ind_matrix, const DoubleComplex* values_matrix, DoubleComplex* d_inverse)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid < N)
    {
        int flag = -1;
        int row = gid;
        int start_index = (row_ptr_matrix)[row];
        int end_index = (row_ptr_matrix)[row + 1];

        for (int i = start_index; i < end_index; i++)
        {
            if ((col_ind_matrix)[i] == row)
            {
                if ((values_matrix)[i] != 0)
                {

                    flag = 0;
                    // double re = real((csr_mat->gpu_values)[i]);
                    d_inverse[row] = 1.0 / (values_matrix)[i];
                }
                break;
            }
        }

        if (flag == -1)
        {
            printf("\nJacobi preconditioner generation failed!\n");
            // __threadfence();
            // asm("trap;");

            // [font = "Courier New"] __threadfence()[/ font];
            // [font = "Courier New"] asm("trap; ")[/ font];
            assert(0);
        }
    }
}


//! Initializes the Jacobi Preconditioner internal array(on GPU) based on the input sparse matrix using a GPU kernel
/*!
   \param[out] A reference to CSR Matrix object
   \return error code 0 if the initialization is successful else returns error code = -1
*/
int JacobiPreconditioner::Initialize_Preconditioner(const CSR_Matrix& A)
{
    if (Exists_gpu() == false)
        Allocate_Memory(LOCATION::GPU);

    int err_code = 0;
    int N = Get_Diag_Length();
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil((double)N / (double)THREADS_PER_BLOCK));
    Diagonal_Scaling_Jacobi << < grid, block >> > (N, A.GetGPURowPtr(), A.GetGPUColInd(), A.GetGPUValues(), Get_GPU_d_inverse());
    // printf("\n");
     // printf(cudaGetErrorString(cudaDeviceSynchronize()));
    // printf("\n");
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        err_code = -1;
    }

    return err_code; //0 for successful preconditioner generation , return -1 if it fails.

}


//! Parameterized constructor for Jacobi Preconditioner class
/*!
  Generates jacobi preconditioner based on the input sparse matrix 
  It exits the program, in case, the generation fails.
 \param[in] A refernce to CSR Matrix object
*/
JacobiPreconditioner::JacobiPreconditioner(const CSR_Matrix& A) : Preconditioner(PRECONDITIONER_TYPE::JACOBI), diag_length{ A.GetRows() }
{

    Allocate_Memory(LOCATION::GPU);
    int err_code = Initialize_Preconditioner(A);
    if (err_code != 0)
    {
        std::cout << "\nError while initilizing preconditioner\n";
        exit(1);
    }
    else
    {
        std::cout << "\nJacobi preconditioner generated successfully!\n";
    }

}


//! Allocates memory for Jacobi Preconditioner object's internel array on the specified location
/*!
\param[in] loc enum type variable which indicates the location(CPU/GPU) where the memory is to be allocated
*/
void JacobiPreconditioner::Allocate_Memory(const LOCATION loc)
{
    if (loc == LOCATION::GPU && Exists_gpu()==false)
    {
        cudaMalloc((void**)&gpu_d_inverse, Get_Diag_Length() * sizeof(DoubleComplex));
        this->gpu_exists = GPU_EXISTENCE::EXISTENT;
    }
    else if (loc == LOCATION::CPU && Exists_cpu()==true)
    {
        cpu_d_inverse = new DoubleComplex[Get_Diag_Length()];
        this->cpu_exists = CPU_EXISTENCE::EXISTENT;
    }
}


//! Deallocates resources of the preconditioner object 
/*!
 \param[in] loc enum type variable which indicates the location, where the resources are to be deallocated
*/
void JacobiPreconditioner::Deallocate_Memory(const LOCATION loc)
{
    if (loc == LOCATION::GPU && Exists_gpu() == true)
    {
        cudaFree(gpu_d_inverse);
        this->gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
        gpu_d_inverse = nullptr;
    }
    else if (loc == LOCATION::CPU && Exists_cpu() == true)
    {
        delete[] cpu_d_inverse;
        this->cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
        cpu_d_inverse = nullptr;
    }
}



//! Destructor for JacobiPreconditioner class
/*!
  It deallocates acquires resources, if any.
*/
JacobiPreconditioner:: ~JacobiPreconditioner()
{
    if (Exists_gpu() == true)
        Deallocate_Memory(LOCATION::GPU);
    else if (Exists_cpu() == true)
        Deallocate_Memory(LOCATION::CPU);
}

//! Copies jacobi preconditioner object's internal arrays from CPU to GPU memory
void JacobiPreconditioner::CopyPreconditioner_cpu_to_gpu()
{ 
    assert(Exists_cpu() == true);
    if (Exists_gpu() == false)
        Allocate_Memory(LOCATION::GPU);
    cudaMemcpy(Get_GPU_d_inverse(), Get_CPU_d_inverse(), Get_Diag_Length() * sizeof(DoubleComplex), cudaMemcpyHostToDevice);

}

//! Copies jacobi preconditioner object's internal arrays from GPU to CPU memory
void JacobiPreconditioner::CopyPreconditioner_gpu_to_cpu()
{
    assert(Exists_gpu() == true);
    if (Exists_cpu() == false)
        Allocate_Memory(LOCATION::CPU);
    cudaMemcpy(Get_CPU_d_inverse(), Get_GPU_d_inverse(), Get_Diag_Length() * sizeof(DoubleComplex), cudaMemcpyDeviceToHost);
}


//! Generates a preconditioner based on the preconditioner type and sparse matrix received as input
/*!
  \param[in] precond_type enum varaible which describes the type of the preconditioner
  \param[in] A reference to CSR matrix object
  \return a pointer to the generated preconditioner
*/
Preconditioner* Generate_Preconditioner(const PRECONDITIONER_TYPE precond_type, const CSR_Matrix& A)
{
    Preconditioner* precond = nullptr;
    switch (precond_type)
    {
    case PRECONDITIONER_TYPE::JACOBI:
        precond = new JacobiPreconditioner(A);
        break;

    case PRECONDITIONER_TYPE::RICHARDSON:
        precond = new RichardsonPreconditioner(A);
        break;    
    }

    return precond;
}


//! GPU kernel which performs the jacobi preconditioning operation
/*!
  Each element of the result is handled using a different thread. All elements are computed in parallel.   
  result = jacobiPreconditioner(d_inverse) * vector
 \param[in] N length of vector
 \param[in] d_inverse base address of array allocated on GPU which stores jacobi preconditioner values
 \param[in] vec the array allocated on GPU with which preconditioner is multiplied
 \param[out] result the base address of the array allocated on GPU where result of mutiplication of preconditioner and the vector
*/
__global__ void Jacobi_precond_matrix_vector_multiplication(const int N, const DoubleComplex* d_inverse, const DoubleComplex* vec, DoubleComplex* result)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid < N)
    {
        result[gid] = d_inverse[gid] * vec[gid];
    }
}


//! Performs jacobi preconditioning operation using GPU kernel
/*!
  \param[in] vec reference to a dense matrix object( a vector)
  \param[out] result formed by result = jacobi_preconditioner matrix(d_inverse) * vec

   Operates on internal GPU arrays.
 */
void JacobiPreconditioner::ApplyPreconditioner(const Dense_Matrix& vec, Dense_Matrix& result) const {
    
    assert(this->Exists_gpu() == true);
    assert(vec.ExistsGPU() == true);
    assert(result.ExistsGPU() == true);
    dim3 block(THREADS_PER_BLOCK);
    int N = Get_Diag_Length();
    dim3 grid(ceil((double)N / (double)THREADS_PER_BLOCK));
    Jacobi_precond_matrix_vector_multiplication << < grid, block >> > (N, Get_GPU_d_inverse(), vec.GetGPUValues(), result.GetGPUValues()); 
}



//------------------------------------------------------------------------------------------------------------------------
                                /* Functions amd kernels for RichardsonPreconditioner class */


//! Parameterized constructor for Richardson Preconditioner class
/*!
  Generates richardson preconditioner based on the input sparse matrix.
  The preconditioner values are not explicitly stored as it is an Identity matrix
 \param[in] A reference to CSR Matrix object
*/
RichardsonPreconditioner::RichardsonPreconditioner(const CSR_Matrix& A)
: Preconditioner(PRECONDITIONER_TYPE::RICHARDSON), Identity_dim{ A.GetRows() }
{
   
}



//! A GPU kernel which performs richardson preconditioning operation
/*! 
   Each element of the result is handled using a different thread. All elements are computed in parallel. 
 \param[in] N length of vector
 \param[in] vec array allocated on GPU storing vector values
 \param[out] result array allocated on GPU where the result of the preconditioning operation is to stored

*/
__global__ void Richardson_Precond_applicn(const int N , const DoubleComplex* vec , DoubleComplex* result)
{
  int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid < N)
    {
        result[gid] = vec[gid];
    }
}


//! Performs richardson preconditioning operation using a GPU kernel
/*!
 \param[in] vec  reference to a dense matrix object(vector)
 \param[out] result  reference to a dense matrix object which is going to store the result of preconditioning

   Operates on internal GPU arrays.
*/  
 void RichardsonPreconditioner::ApplyPreconditioner(const Dense_Matrix& vec, Dense_Matrix& result) const {
  
  assert(vec.ExistsGPU() == true);
  assert(result.ExistsGPU() == true);
  dim3 block(THREADS_PER_BLOCK);
  int N = Get_Identity_Dimension();
  dim3 grid(ceil(static_cast<double>(N) / static_cast<double>(THREADS_PER_BLOCK)));
  Richardson_Precond_applicn <<< grid , block>>>(N,vec.GetGPUValues(),result.GetGPUValues());
 
}



