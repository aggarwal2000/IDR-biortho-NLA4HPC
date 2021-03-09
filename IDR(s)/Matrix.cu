/*!
\file Matrix.cu
 \brief Implementation of member functions of classes Dense_Matrix, CSR_Matrix and COO_Matrix
*/

#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
#include "Matrix.h"
#include "kernels.h"
#include "header.h"



//----------------------------------------------------------------------------------------------------------------------
/* Member functions for Dense matrix class*/



//! Move assignment operator for Dense Matrix class
/*!
   It moves the CPU and GPU resources/state of the temporary object(input) into the current class object.
   \param[in,out] mat rvalue reference which binds to dense matrix class temporary object
   \return reference(lvalue) to the current object
*/
Dense_Matrix& Dense_Matrix::operator= (Dense_Matrix&& mat)
{
	assert(GetRows() == mat.GetRows());
	assert(GetCols() == mat.GetCols());
	assert(GetLda() == mat.GetLda());
	assert(GetOrder() == mat.GetOrder());

	cpu_exists = mat.cpu_exists;
	cpu_values = mat.cpu_values;
	mat.cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
	mat.cpu_values = nullptr;

	gpu_exists = mat.gpu_exists;
	gpu_values = mat.gpu_values;
	mat.gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
	mat.gpu_values = nullptr;

	return *this;
}


//! Copy assignment operator for dense matrix class
/*!
  This copies the resources/state of the input object(lvalue reference) to the current class object.
  \param[in] mat lvalue reference which binds to an lvalue -dense matrix class object
  \return reference(lvalue) to the current object
*/
Dense_Matrix& Dense_Matrix::operator= (const Dense_Matrix& mat)
{
	assert(GetRows() == mat.GetRows());
	assert(GetCols() == mat.GetCols());
	assert(GetLda() == mat.GetLda());
	assert(GetOrder() == mat.GetOrder());
	if (mat.ExistsCPU() == true)
	{
		if (ExistsCPU() == false)
			Allocate_Memory(LOCATION::CPU);
		Copy_array_cpu_to_cpu(mat.GetCPUValues(), GetCPUValues(), GetLda() * GetCols());
	}
	else
		Deallocate_Memory(LOCATION::CPU);

	if (mat.ExistsGPU() == true)
	{
		if (ExistsGPU() == false)
			Allocate_Memory(LOCATION::GPU);
		Copy_array_gpu_to_gpu(mat.GetGPUValues(), GetGPUValues(), GetLda() * GetCols());
	}
	else
		Deallocate_Memory(LOCATION::GPU);

	return *this;
}


//! Move constructor for dense matrix class
/*!
   Forms a dense matrix object by moving CPU and GPU resources/state of a temporary object into it
   \param[in,out] mat rvalue reference which binds to dense matrix class temporary object
*/
Dense_Matrix::Dense_Matrix(Dense_Matrix&& mat)
	: rows{ mat.rows }, cols{ mat.cols }, lda{ mat.lda }, order{ mat.order }
{

	cpu_exists = mat.cpu_exists;
	cpu_values = mat.cpu_values;
	mat.cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
	mat.cpu_values = nullptr;

	gpu_exists = mat.gpu_exists;
	gpu_values = mat.gpu_values;
	mat.gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
	mat.gpu_values = nullptr;

}


//! Copy constructor for dense matrix class
/*!

   Forms a dense matrix object by copying CPU and GPU resources/state of the input(lvalue reference) into it
   \param[in] mat lvalue reference which binds to an lvalue -dense matrix class object
*/
Dense_Matrix::Dense_Matrix(const Dense_Matrix& mat)
	: rows{ mat.rows }, cols{ mat.cols }, lda{ mat.lda }, order{ mat.order }
{
	if (mat.ExistsCPU() == true)
	{
		Allocate_Memory(LOCATION::CPU);
		Copy_array_cpu_to_cpu(mat.cpu_values, this->cpu_values, lda * cols);
	}

	if (mat.ExistsGPU() == true)
	{
		Allocate_Memory(LOCATION::GPU);
		Copy_array_gpu_to_gpu(mat.gpu_values, this->gpu_values, lda * cols);
	}
}



//! Allocates memory on the specified location to store dense matirx's internal ararys 
/*!
 Allocates memory  on the specified location based on the leading dimension and the number of colums of the dense matrix object
\param[in] loc enum type which indicates the location -either GPU or CPU, where the dense matrix values are to be stored
*/
void Dense_Matrix::Allocate_Memory(const LOCATION loc)
{
	if (loc == LOCATION::CPU && ExistsCPU() == false)
	{
		cpu_values = new DoubleComplex[GetLda() * GetCols()];
		cpu_exists = CPU_EXISTENCE::EXISTENT;
	}
	else if (loc == LOCATION::GPU && ExistsGPU() == false)
	{
		cudaMalloc((void**)&gpu_values, GetLda() * GetCols() * sizeof(DoubleComplex));
		gpu_exists = GPU_EXISTENCE::EXISTENT;
	}
}



//! A parameterized constructor for dense matrix class
/*!
 \param[in] num_rows number of rows in dense matrix object being formed
 \param[in] num_cols number of columns in dense matrix object being formed
 \param[in] lda_mat leading dimension of the dense matrix object being formed.(Usually the number of rows is rounded up to a certain value to give lda.)
 \param[in] order_mat storage order of the dense matrix object being formed
 \param[in] cpu_exists enum type variable which indicates if memory is to be allocated on CPU for the object's internals(the values array) being formed
 \param[in] gpu_exists enum type variable which indicates if memory is to be allocated on GPU for the object's internals(the values array) being formed
*/
Dense_Matrix::Dense_Matrix(const int num_rows, const int num_cols, const int lda_mat, const ORDER order_mat, const CPU_EXISTENCE cpu_exists, const GPU_EXISTENCE gpu_exists)
	: rows{ num_rows }, cols{ num_cols }, lda{ lda_mat }, order{ order_mat }
{
    assert(lda >= rows);
	if (cpu_exists == CPU_EXISTENCE::EXISTENT)
		Allocate_Memory(LOCATION::CPU);

	if (gpu_exists == GPU_EXISTENCE::EXISTENT)
		Allocate_Memory(LOCATION::GPU);

}



//! Destructor for dense matrix class object
/*!
   Called automatically when the dense matrix object is destroyed. It deallocates the acquired resources, if any.
*/
Dense_Matrix::~Dense_Matrix()
{
	if (ExistsCPU() == true)
		Deallocate_Memory(LOCATION::CPU);

	if (ExistsGPU() == true)
		Deallocate_Memory(LOCATION::GPU);
}


//! Copies dense matrix values from CPU to GPU
/*!
  Allocates memory on GPU if required. Copies all values from CPU memory to GPU.
*/
void Dense_Matrix::CopyMatrix_cpu_to_gpu()
{   
	assert(ExistsCPU() == true);
	if (ExistsGPU() == false)
		Allocate_Memory(LOCATION::GPU);

	cudaMemcpy(GetGPUValues(), GetCPUValues(), GetLda() * GetCols() * sizeof(DoubleComplex), cudaMemcpyHostToDevice);

}


//! Copies dense matrix values from GPU to CPU
/*!
  Allocates memory on CPU if required. Copies all values from GPU memory to CPU.
*/
void Dense_Matrix::CopyMatrix_gpu_to_cpu()
{
	assert(ExistsGPU() == true);
	if (ExistsCPU() == false)
		Allocate_Memory(LOCATION::CPU);

	cudaMemcpy(GetCPUValues(), GetGPUValues(), GetLda() * GetCols() * sizeof(DoubleComplex), cudaMemcpyDeviceToHost);
}


//! Copies a part of a dense matrix values from CPU to GPU
/*!
  Allocates memory on GPU if required. Copies a part of matrix defined by starting and ending column and row indices from CPU memory to GPU memory
 \param[in] col_start index of the starting column
 \param[in] col_end index of the ending column
 \param[in] row_start index of the starting row
 \param[in] row_end index of the ending row
*/
void Dense_Matrix::CopyMatrix_cpu_to_gpu(int col_start, int col_end, int row_start, int row_end)//write version for submatrix -copy
{ 
	assert(ExistsCPU() == true);
	if (ExistsGPU() == false)
		Allocate_Memory(LOCATION::GPU);

	int N = row_end - row_start + 1;

	DoubleComplex* cpu_val, * gpu_val;

	for (int i = col_start; i <= col_end; i++)
	{
		cpu_val = GetSpecificLocationPtrCPU(row_start, i);
		gpu_val = GetSpecificLocationPtrGPU(row_start, i);

		cudaMemcpy(gpu_val, cpu_val, N * sizeof(DoubleComplex), cudaMemcpyHostToDevice);

	}

}



//! Copies a part of a dense matrix values from GPU to CPU
/*!
  Allocates memory on CPU if required. Copies a part of matrix defined by starting and ending column and row indices from GPU memory to CPU memory
 \param[in] col_start index of the starting column
 \param[in] col_end index of the ending column
 \param[in] row_start index of the starting row
 \param[in] row_end index of the ending row
*/
void Dense_Matrix::CopyMatrix_gpu_to_cpu(int col_start, int col_end, int row_start, int row_end)
{
	if (ExistsCPU() == false)
		Allocate_Memory(LOCATION::CPU);

	int N = row_end - row_start + 1;

	DoubleComplex* cpu_val, * gpu_val;

	for (int i = col_start; i <= col_end; i++)
	{
		cpu_val = GetSpecificLocationPtrCPU(row_start, i);
		gpu_val = GetSpecificLocationPtrGPU(row_start, i);

		cudaMemcpy(cpu_val, gpu_val, N * sizeof(DoubleComplex), cudaMemcpyDeviceToHost);

	}
}


//! Deallocates specified location's resources of the dense matrix object 
/*!
 \param[in] loc enum type varaible which indicates the location(CPU/GPU) where the resources are to be dealloacted 
*/
void Dense_Matrix::Deallocate_Memory(const LOCATION loc)
{
	if (loc == LOCATION::CPU && ExistsCPU() == true)
	{
		delete[] cpu_values;
		cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
		cpu_values = nullptr;
	}
	if (loc == LOCATION::GPU && ExistsGPU() == true)
	{
		cudaFree(gpu_values);
		gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
		gpu_values = nullptr;
	}
}




//-------------------------------------------------------------------------------------------------------------
/* CSR Matrix member functions */


//! Allocates memory on the specified location to store CSR matirx's internal ararys 
/*!
  Allocates memory based on the number of rows and non zero elements of the CSR matrix object
\param[in] loc enum type which indicates the location -either GPU or CPU, where the CSR matrix related arrays are to be stored
*/
void CSR_Matrix::Allocate_Memory(const LOCATION loc)
{
	if (loc == LOCATION::CPU && ExistsCPU() == false)
	{
		cpu_values = new DoubleComplex[Getnz()];
		cpu_row_ptr = new int[GetRows() + 1];
		cpu_col_ind = new int[Getnz()];
		cpu_exists = CPU_EXISTENCE::EXISTENT;
	}
	else if (loc == LOCATION::GPU && ExistsGPU() == false)
	{
		cudaMalloc((void**)&gpu_values, Getnz() * sizeof(DoubleComplex));
		cudaMalloc((void**)&gpu_row_ptr, (GetRows() + 1) * sizeof(int));
		cudaMalloc((void**)&gpu_col_ind, Getnz() * sizeof(int));
		gpu_exists = GPU_EXISTENCE::EXISTENT;
	}
}



//! A parameterized constructor for CSR matrix class
/*!
 \param[in] num_rows number of rows in CSR matrix object being formed
 \param[in] num_cols number of columns in CSR matrix object being formed
 \param[in] nz_mat number of non zero elements in the CSR matrix object being formed
 \param[in] cpu_exists enum type variable which indicates if memory is to be allocated on CPU for the object's internals(CSR matrix arrays) being formed
 \param[in] gpu_exists enum type variable which indicates if memory is to be allocated on GPU for the object's internals(CSR matrix arrays) being formed
*/
CSR_Matrix::CSR_Matrix(const int num_rows, const int num_cols, const int nz_mat, const CPU_EXISTENCE cpu_exists, const GPU_EXISTENCE gpu_exists)
	: rows{ num_rows }, cols{ num_cols }, nz{ nz_mat }
{

	if (cpu_exists == CPU_EXISTENCE::EXISTENT)
		Allocate_Memory(LOCATION::CPU);
	if (gpu_exists == GPU_EXISTENCE::EXISTENT)
		Allocate_Memory(LOCATION::GPU);

}


//! Destructor for CSR matrix class object
/*!
   Called automatically when the CSR matrix object is destroyed. It deallocates the acquired resources, if any.
*/
CSR_Matrix::~CSR_Matrix()
{
	if (ExistsCPU() == true)
		Deallocate_Memory(LOCATION::CPU);

	if (ExistsGPU() == true)
		Deallocate_Memory(LOCATION::GPU);
}


//! Deallocates specified location's resources of the CSR matrix object 
/*!
 \param[in] loc enum type varaible which indicates the location(CPU/GPU) where the resources are to be dealloacted 
*/
void CSR_Matrix::Deallocate_Memory(const LOCATION loc)
{
	if (loc == LOCATION::CPU && ExistsCPU() == true)
	{
		delete[] cpu_values;
		delete[] cpu_col_ind;
		delete[] cpu_row_ptr;
		cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
		cpu_values = nullptr;
		cpu_col_ind = nullptr;
		cpu_row_ptr = nullptr;
	}

	if (loc == LOCATION::GPU && ExistsGPU() == true)
	{
		cudaFree(gpu_values);
		cudaFree(gpu_row_ptr);
		cudaFree(gpu_col_ind);
		gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
		gpu_values = nullptr;
		gpu_row_ptr = nullptr;
		gpu_col_ind = nullptr;
	}
}


//! Copies CSR matrix internal arrays from CPU to GPU
/*!
    Allocates memory on GPU if required. Copies all internal arrays from CPU memory to GPU.
*/
void  CSR_Matrix::CopyMatrix_cpu_to_gpu()
{  
	assert(ExistsCPU() == true);
	if (ExistsGPU() == false)
		Allocate_Memory(LOCATION::GPU);
	cudaMemcpy(GetGPUValues(), GetCPUValues(), Getnz() * sizeof(DoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(GetGPURowPtr(), GetCPURowPtr(), (GetRows() + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GetGPUColInd() , GetCPUColInd(), Getnz() * sizeof(int), cudaMemcpyHostToDevice);
}


//! Copies CSR matrix internal arrays from GPU to CPU
/*!
    Allocates memory on CPU if required. Copies all internal arrays from GPU memory to CPU.
*/
void CSR_Matrix::CopyMatrix_gpu_to_cpu()
{  
	assert(ExistsGPU() == true);
	if (ExistsCPU() == false)
		Allocate_Memory(LOCATION::CPU);
	cudaMemcpy(GetCPUValues() , GetGPUValues() , Getnz() * sizeof(DoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(GetCPURowPtr(), GetGPURowPtr() , (GetRows() + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy( GetCPUColInd(), GetGPUColInd(), Getnz() * sizeof(int), cudaMemcpyDeviceToHost);
}



//-------------------------------------------------------------------------------------------------------------------
  /* Member functions for COO matrix class */


//! Allocates memory on the specified location to store COO matirx internal ararys 
/*!
  Allocates memory based on the number of non zero elements of the COO matrix object
\param[in] loc enum type which indicates the location -either GPU or CPU, where the COO matrix related arrays are to be stored
*/
void COO_Matrix::Allocate_Memory(const LOCATION loc)
{
	if (loc == LOCATION::CPU && ExistsCPU() == false)
	{
		cpu_values = new DoubleComplex[Getnz()];
		cpu_row_ind = new int[Getnz()];
		cpu_col_ind = new int[Getnz()];
		cpu_exists = CPU_EXISTENCE::EXISTENT;
	}
	else if (loc == LOCATION::GPU && ExistsGPU()== false)
	{
		cudaMalloc((void**)&gpu_values, Getnz() * sizeof(DoubleComplex));
		cudaMalloc((void**)&gpu_row_ind, Getnz() * sizeof(int));
		cudaMalloc((void**)&gpu_col_ind, Getnz() * sizeof(int));
		gpu_exists = GPU_EXISTENCE::EXISTENT;
	}
}


//! A parameterized constructor for COO matrix class
/*!
 \param[in] num_rows number of rows in COO matrix object being formed
 \param[in] num_cols number of columns in COO matrix object being formed
 \param[in] mat_nz number of non zero elements in the COO matrix object being formed
 \param[in] cpu_exists enum type variable which indicates if memory is to be allocated on CPU for the object's internals(COO matrix arrays) being formed
 \param[in] gpu_exists enum type variable which indicates if memory is to be allocated on GPU for the object's internals(COO matrix arrays) being formed
*/
COO_Matrix::COO_Matrix(const int num_rows, const int num_cols, const int mat_nz, const CPU_EXISTENCE cpu_exists, const GPU_EXISTENCE gpu_exists)
	: rows{ num_rows }, cols{ num_cols }, nz{ mat_nz }
{
	
	if (cpu_exists == CPU_EXISTENCE::EXISTENT)
		Allocate_Memory(LOCATION::CPU);
	if (gpu_exists == GPU_EXISTENCE::EXISTENT)
		Allocate_Memory(LOCATION::GPU);

}


//! Destructor for COO matrix class object
/*!
   Called automatically when the COO matrix object is destroyed. It deallocates the acquired resources, if any.
*/
COO_Matrix::~COO_Matrix()
{
	if (ExistsCPU() == true)
		Deallocate_Memory(LOCATION::CPU);

	if (ExistsGPU() == true)
		Deallocate_Memory(LOCATION::GPU);
}


//! Deallocates specified location's resources of the COO matrix object 
/*!
 \param[in] loc enum type varaible which indicates the location(CPU/GPU) where the resources are to be dealloacted 
*/
void COO_Matrix::Deallocate_Memory(const LOCATION loc)
{
	if (loc == LOCATION::CPU && ExistsCPU() == true)
	{
		delete[] cpu_values;
		delete[] cpu_col_ind;
		delete[] cpu_row_ind;
		cpu_exists = CPU_EXISTENCE::NON_EXISTENT;
		cpu_values = nullptr;
		cpu_col_ind = nullptr;;
		cpu_row_ind = nullptr;

	}

	if (loc == LOCATION::GPU && ExistsGPU() == true)
	{
		cudaFree(gpu_values);
		cudaFree(gpu_row_ind);
		cudaFree(gpu_col_ind);
		gpu_exists = GPU_EXISTENCE::NON_EXISTENT;
		gpu_values = nullptr;
		gpu_row_ind = nullptr;
		gpu_col_ind = nullptr;
	}
}


//! Copies COO matrix internal arrays from CPU to GPU
/*!
    Allocates memory on GPU if required. Copies all internal arrays from CPU memory to GPU.
*/
void COO_Matrix::CopyMatrix_cpu_to_gpu()
{ 
	assert(ExistsCPU() == true);
	if (ExistsGPU() == false)
		Allocate_Memory(LOCATION::GPU);
	cudaMemcpy(GetGPUValues(), GetCPUValues(), Getnz() * sizeof(DoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(GetGPURowInd(), GetCPURowInd(), Getnz() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GetGPUColInd(), GetCPUColInd(), Getnz() * sizeof(int), cudaMemcpyHostToDevice);
}


//! Copies COO matrix internal arrays from GPU to CPU
/*!
    Allocates memory on CPU if required. Copies all internal arrays from GPU memory to CPU.
*/
void COO_Matrix::CopyMatrix_gpu_to_cpu()
{
	assert(ExistsGPU() == true);
	if (ExistsCPU() == false)
		Allocate_Memory(LOCATION::CPU);
	cudaMemcpy(GetCPUValues(), GetGPUValues(), Getnz() * sizeof(DoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(GetCPURowInd(), GetGPURowInd() , Getnz() * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(GetCPUColInd(), GetGPUColInd(), Getnz() * sizeof(int), cudaMemcpyDeviceToHost);
}