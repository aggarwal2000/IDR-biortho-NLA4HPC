/*!
\file Matrix.h
 \brief  Definition of classes Dense_Matrix, CSR_Matrix and COO_Matrix
*/
# pragma once

#include<cassert>

#include<cuComplex.h>

#include"location_enums.h"

typedef cuDoubleComplex DoubleComplex;



//! enum class which defines storage order for dense matrices
/**
 This enum class defines the order(row/column) in which values are stored in dense matrices
*/
enum class ORDER {
	COLUMN_MAJOR, /*!< means that the matrix values are stored in column major order */
	ROW_MAJOR  /*!< means that the matrix values are stored in row major order */
};





//! Class for Complex Dense Matrices
/*!
    This class contains attributes and member functions for complex dense matrix class objects.
*/
class Dense_Matrix {
private:
	const int rows; /*!< Number of rows in dense matrix*/
	const int cols; /*!< Number of columns in dense matrix */
	const int lda; /*!< Leading Dimension of dimension of dense matrix (Usually the number of rows is rounded up to a certain value to give lda.) */
	ORDER order = ORDER::COLUMN_MAJOR; /*!< storage order of dense matrix */
	CPU_EXISTENCE cpu_exists = CPU_EXISTENCE::NON_EXISTENT; /*!< presence/absence of dense matrix internals(large arrays) in CPU memory*/
	GPU_EXISTENCE gpu_exists = GPU_EXISTENCE::NON_EXISTENT; /*!< presence/absence of dense matrix internals(large arrays) in GPU memory*/
	DoubleComplex* cpu_values = nullptr; /*!< Pointer storing base address of the array allocated on CPU containing dense matrix's values.
	 It is equal to nullptr in case no memory is allocated. Note:values on CPU and GPU do not match until they are copied.*/
	DoubleComplex* gpu_values = nullptr;/*!< Pointer storing base address of the array allocated on GPU containing dense matrix's values.
	 It is equal to nullptr in case no memory is allocated. Note:values on CPU and GPU do not match until they are copied.*/
public:

	//! Returns number of rows in dense matrix
	/*!
	   \return number of rows in dense matrix
	*/
	int GetRows() const
	{
		return rows;
	}


    //! Returns number of columns in dense matrix
    /*!
     \return number of columns in dense matrix
    */
	int GetCols() const
	{
		return cols;
	}


    //! Returns leading dimension of dense matrix
    /*!
    \return leading dimension of dense matrix
    */
	int GetLda() const
	{
		return lda;
	}


    //! Returns storage order of dense matrix
    /*!
    \return order in which values are stored in the dense matrix
    */
	ORDER GetOrder() const
	{
		return order;
	}




    //! Returns a pointer to the array allocated on CPU which stores the dense matrix values
    /*!
     \return pointer to the array allocated on CPU which stores the dense matrix values; nullptr in case no such array exists 
    */
	DoubleComplex* GetCPUValues() const
	{   
		if(ExistsCPU() == true)
		  return cpu_values;
		else
			return nullptr;
	}

    
     //! Returns a pointer to the array allocated on GPU which stores the dense matrix values
    /*!
     \return pointer to the array allocated on GPU which stores the dense matrix values; nullptr in case no such array exists 
    */
	DoubleComplex* GetGPUValues() const
	{   
		if(ExistsGPU() == true)
		  return gpu_values;
		else
			return nullptr;
	}



    //! Takes in a column index and returns a pointer to its starting location on GPU. 
    /*!
    \param[in] col_ind index of the column for which the pointer (to GPU memory) is required
    \return pointer to the starting location of the column on GPU memory; nullptr in case no memory is allocated on GPU
    */
	DoubleComplex* GetColPtrGPU(const int col_ind) const
	{
		assert(col_ind < GetCols());
		//return &gpu_values[lda * col_ind];
		if(ExistsGPU() == true)
		   return GetGPUValues() + GetLda() * col_ind;
		else
			return nullptr;
	}



    //! Takes in a column index and returns a pointer to its starting location on CPU. 
    /*!
    \param[in] col_ind index of the column for which the pointer (to CPU memory) is required
    \return pointer to the starting location of the column on CPU memory; nullptr in case no memory is allocated on CPU
    */
	DoubleComplex* GetColPtrCPU(const int col_ind) const
	{   
		assert(col_ind < GetCols());
		if(ExistsCPU() == true)
		  return  GetCPUValues() +  GetLda() * col_ind;
		else
			return nullptr;
	}


   //! Takes in a location and returns its GPU memory address
	/*!
	\param[in] row row index of the element 
	\param[in] col column index the element
	\return pointer(GPU memory address) to the element; nullptr in case no memory is allocated on GPU
	*/
	DoubleComplex* GetSpecificLocationPtrGPU(const int row, const int col) const
	{
		//return &gpu_values[row + lda * col];
		assert(row < GetRows());
		assert(col < GetCols());
		if(ExistsGPU() == true)
		  return GetGPUValues() + row +  GetLda()* col;
		else
			return nullptr;
	}


    //! Takes in a location and returns its CPU memory address
	/*!
	\param[in] row row index of the element 
	\param[in] col column index the element
	\return pointer(CPU memory address) to the element; nullptr in case no memory is allocated on CPU
	*/
	DoubleComplex* GetSpecificLocationPtrCPU(const int row, const int col) const
	{   
		assert(row < GetRows());
		assert(col < GetCols());
		if(ExistsCPU() == true)
		  return GetCPUValues() +  row +  GetLda()* col;
		else
			return nullptr;
	}


    //! Returns true if Dense matrix values are present on CPU memory
    /*!
    \return boolean value
    */
	bool ExistsCPU() const
	{
		return cpu_exists == CPU_EXISTENCE::EXISTENT;
	}


     //! Returns true if Dense matrix values are present on GPU memory
    /*!
    \return boolean value
    */
	bool ExistsGPU() const
	{
		return gpu_exists == GPU_EXISTENCE::EXISTENT;
	}

    

	void Allocate_Memory(const LOCATION loc);

	Dense_Matrix(const int rows, const int cols, const int lda, const ORDER order, const CPU_EXISTENCE cpu_exists, const GPU_EXISTENCE gpu_exists);

	~Dense_Matrix();

	void CopyMatrix_cpu_to_gpu();

	void CopyMatrix_gpu_to_cpu();

	void Deallocate_Memory(const LOCATION loc);

	Dense_Matrix(const Dense_Matrix& mat);

	Dense_Matrix(Dense_Matrix&& mat);

	Dense_Matrix& operator= (const Dense_Matrix& mat);

	Dense_Matrix& operator= (Dense_Matrix&& mat);

	void CopyMatrix_cpu_to_gpu(int col_start , int col_end , int row_start , int row_end);

	void CopyMatrix_gpu_to_cpu(int col_start , int col_end, int row_start, int row_end);

};







//! Class for complex sparse CSR matrix
/*!
    This class contains attributes and member functions for complex sparse CSR matrix class.
*/
class CSR_Matrix {
private:
	const int rows; /*!< Number of rows in CSR matrix*/
	const int cols; /*!< Number of columns in CSR matrix*/
	const int nz;  /*!< Number of nonzero elemnents in CSR matrix*/
	CPU_EXISTENCE cpu_exists = CPU_EXISTENCE::NON_EXISTENT; /*!< presence/absence of CSR matrix internals in CPU memory*/
	GPU_EXISTENCE gpu_exists = GPU_EXISTENCE::NON_EXISTENT;  /*!< presence/absence of CSR matrix internals in GPU memory*/
	DoubleComplex* cpu_values = nullptr; /*!< Pointer storing base address of the array allocated on CPU containing CSR matrix's values.
	 It is equal to nullptr in case no memory is allocated. Note:values on CPU and GPU do not match until they are copied.*/
	DoubleComplex* gpu_values = nullptr;  /*!< Pointer storing base address of the array allocated on GPU containing CSR matrix's values.
	 It is equal to nullptr in case no memory is allocated. Note:values on CPU and GPU do not match until they are copied.*/
	int* cpu_row_ptr = nullptr; /*!< Pointer storing base address of the array allocated on CPU containing CSR matrix row pointers 
	 It is equal to nullptr in case no memory is allocated.*/
	int* gpu_row_ptr = nullptr;/*!< Pointer storing base address of the array allocated on GPU containing CSR matrix row pointers 
	 It is equal to nullptr in case no memory is allocated.*/
	int* cpu_col_ind = nullptr;/*!< Pointer storing base address of the array allocated on CPU containing CSR matrix column indices 
	 It is equal to nullptr in case no memory is allocated.*/
	int* gpu_col_ind = nullptr;/*!< Pointer storing base address of the array allocated on GPU containing CSR matrix column indices 
	 It is equal to nullptr in case no memory is allocated.*/
public:

	//! Returns number of rows in CSR matrix
	/*!
	   \return number of rows in CSR matrix
	*/
	int GetRows() const
	{
		return rows;
	}
    

    //! Returns number of columns in CSR matrix
	/*!
	   \return number of columns in CSR matrix
	*/
	int GetCols() const
	{
		return cols;
	}


    //! Returns number of non zero elements in CSR matrix
	/*!
	   \return number of non zero elements in CSR matrix
	*/
	int Getnz() const
	{
		return nz;
	}


    //! Returns a pointer to the array allocated on CPU which stores the CSR matrix values
    /*!
     \return pointer to the array allocated on CPU which stores the CSR matrix values; nullptr in case no such array exists 
    */
	DoubleComplex* GetCPUValues() const
	{   
		if(ExistsCPU() == true)
		   return cpu_values;
		else
			return nullptr;
	}


    //! Returns a pointer to the array allocated on CPU which stores the CSR matrix row pointers
    /*!
     \return pointer to the array allocated on CPU which stores the CSR matrix row pointers; nullptr in case no such array exists 
    */
	int* GetCPURowPtr() const
	{   
		if(ExistsCPU() == true)
		  return cpu_row_ptr;
		else
		  return nullptr;
	}


    //! Returns a pointer to the array allocated on CPU which stores the CSR matrix column indices
    /*!
     \return pointer to the array allocated on CPU which stores the CSR matrix column indices; nullptr in case no such array exists 
    */
	int* GetCPUColInd() const
	{   
	   if(ExistsCPU() == true)	
		  return cpu_col_ind;
	   else
		  return nullptr;

	}


    //! Returns a pointer to the array allocated on GPU which stores the CSR matrix values
    /*!
     \return pointer to the array allocated on GPU which stores the CSR matrix values; nullptr in case no such array exists 
    */
	DoubleComplex* GetGPUValues() const
	{   
	   if(ExistsGPU() == true)	
		 return gpu_values;
		else
			return nullptr;
	}


    //! Returns a pointer to the array allocated on GPU which stores the CSR matrix row pointers
    /*!
     \return pointer to the array allocated on GPU which stores the CSR matrix row pointers; nullptr in case no such array exists 
    */
	int* GetGPURowPtr() const
	{ 
		if(ExistsGPU() == true)
		  return gpu_row_ptr;
		else
			return nullptr;
	}


    //! Returns a pointer to the array allocated on GPU which stores the CSR matrix column indices
    /*!
     \return pointer to the array allocated on GPU which stores the CSR matrix column indices; nullptr in case no such array exists 
    */
	int* GetGPUColInd() const
	{
		return gpu_col_ind;
	}


    //! Returns true if CSR matrix internals(large arrays) are present on CPU memory
    /*!
    \return boolean value
    */
	bool ExistsCPU() const
	{
		return cpu_exists == CPU_EXISTENCE::EXISTENT;
	}


    //! Returns true if CSR matrix internals(large arrays) are present on GPU memory
    /*!
    \return boolean value
    */
	bool ExistsGPU() const
	{
		return gpu_exists == GPU_EXISTENCE::EXISTENT;
	}



	void Allocate_Memory(const LOCATION loc);

	CSR_Matrix(const int rows, const int cols, const int nz, const CPU_EXISTENCE cpu_exists, const GPU_EXISTENCE gpu_exists);

	~CSR_Matrix();

	void CopyMatrix_cpu_to_gpu();

	void CopyMatrix_gpu_to_cpu();

	void Deallocate_Memory(const LOCATION loc);

	
    //! Copy constructor for CSR Matrix class
    /*! 
      A deleted constructor
    */
	CSR_Matrix(const CSR_Matrix& mat) = delete;


   
    //! Move constructor for CSR Matrix class
    /*! 
      A deleted constructor
    */
	CSR_Matrix(CSR_Matrix&& mat) = delete;



    //! Copy assignment operator for CSR matrix class
    /*!
        A deleted operator.
    */ 
	CSR_Matrix& operator= (const CSR_Matrix& mat) = delete;


    //! Move assignment operator for CSR matrix class
    /*!
        A deleted operator.
    */ 
	CSR_Matrix& operator= (CSR_Matrix&& mat) = delete;

};




//! Class for complex sparse COO matrix
/*!
  This class contains attributes and member functions for complex COO matrix
*/
class COO_Matrix {
private:
	const int rows;/*!< Number of rows in COO matrix*/
	const int cols;/*!< Number of columns in COO matrix*/
    int nz; /*!< Number of nonzero elemnents in COO matrix, this can also be changed by Set_nz(int) member function*/
	CPU_EXISTENCE cpu_exists = CPU_EXISTENCE::NON_EXISTENT;/*!< presence/absence of COO matrix internals(matrix arrays) in CPU memory*/
	GPU_EXISTENCE gpu_exists = GPU_EXISTENCE::NON_EXISTENT; /*!< presence/absence of COO matrix internals(matrix arrays) in GPU memory*/
	DoubleComplex* cpu_values = nullptr;/*!< Pointer storing base address of the array allocated on CPU containing COO matrix's values.
	 It is equal to nullptr in case no memory is allocated. Note:values on CPU and GPU do not match until they are copied.*/
	DoubleComplex* gpu_values = nullptr;/*!< Pointer storing base address of the array allocated on GPU containing COO matrix's values.
	 It is equal to nullptr in case no memory is allocated. Note:values on CPU and GPU do not match until thay are copied.*/
	int* cpu_row_ind = nullptr;/*!< Pointer storing base address of the array allocated on CPU containing COO matrix row indices 
	 It is equal to nullptr in case no memory is allocated.*/
	int* gpu_row_ind = nullptr;/*!< Pointer storing base address of the array allocated on GPU containing COO matrix row indices 
	 It is equal to nullptr in case no memory is allocated.*/
	int* cpu_col_ind = nullptr;/*!< Pointer storing base address of the array allocated on CPU containing COO matrix column indices 
	 It is equal to nullptr in case no memory is allocated.*/
	int* gpu_col_ind = nullptr;/*!< Pointer storing base address of the array allocated on GPU containing COO matrix column indices 
	 It is equal to nullptr in case no memory is allocated.*/

public:

	//! Returns number of rows in COO matrix
	/*!
	   \return number of rows in COO matrix
	*/
	int GetRows() const
	{
		return rows;
	}
     
    //! Returns number of columns in COO matrix
	/*!
	   \return number of columns in COO matrix
	*/
	int GetCols() const
	{
		return cols;
	}

    //! Returns number of non zero elements in COO matrix
	/*!
	   \return number of non zero elements in COO matrix
	*/
	int Getnz() const
	{
		return nz;
	}

    
    //! Sets the number of non zero elements in COO matrix
	/*!
	   This is used to set the number of non zero elements in COO matrix object after it is formed.
	   Typical example of where it is useful is: If the number of non zero elemnets in COO matrix is unknown in the
	    beginning then an estimate of it is used while creating the object(estimate must be greater else the memory which is
	    allocated based on the estimate won't be enough to store matrix arrays) and later on, after writing the values 
	    into memory and counting the elements along with that, the number of non zero elements can be set to the exact value using this function. 
	   \param[in] mat_nz number of non zero elements to be set for the current matrix object
	*/
	void Set_nz(const int mat_nz)
	{
		nz = mat_nz;
	}

	
    //! Returns a pointer to the array allocated on CPU which stores the COO matrix values
    /*!
     \return pointer to the array allocated on CPU which stores the COO matrix values; nullptr in case no such array exists 
    */
	DoubleComplex* GetCPUValues() const
	{
		return cpu_values;
	}

    //! Returns a pointer to the array allocated on CPU which stores the COO matrix row indices
    /*!
     \return pointer to the array allocated on CPU which stores the COO matrix row indices; nullptr in case no such array exists 
    */
	int* GetCPURowInd() const
	{
		return cpu_row_ind;
	}

    //! Returns a pointer to the array allocated on CPU which stores the COO matrix column indices
    /*!
     \return pointer to the array allocated on CPU which stores the COO matrix column indices; nullptr in case no such array exists 
    */
	int* GetCPUColInd() const
	{
		return cpu_col_ind;

	}


     //! Returns a pointer to the array allocated on GPU which stores the COO matrix values
    /*!
     \return pointer to the array allocated on GPU which stores the COO matrix values; nullptr in case no such array exists 
    */
	DoubleComplex* GetGPUValues() const
	{
		return gpu_values;
	}


     //! Returns a pointer to the array allocated on GPU which stores the COO matrix row indices
    /*!
     \return pointer to the array allocated on GPU which stores the COO matrix row indices; nullptr in case no such array exists 
    */
	int* GetGPURowInd() const
	{
		return gpu_row_ind;
	}


     //! Returns a pointer to the array allocated on GPU which stores the COO matrix column indices
    /*!
     \return pointer to the array allocated on GPU which stores the COO matrix column indices; nullptr in case no such array exists 
    */
	int* GetGPUColInd() const
	{
		return gpu_col_ind;
	}

   

    //! Returns true if COO matrix internals(matrix arrays) are present on CPU memory
    /*!
    \return boolean value
    */
	bool ExistsCPU() const
	{
		return cpu_exists == CPU_EXISTENCE::EXISTENT;
	}


    
    //! Returns true if COO matrix internals(matrix arrays) are present on GPU memory
    /*!
    \return boolean value
    */
	bool ExistsGPU() const
	{
		return gpu_exists == GPU_EXISTENCE::EXISTENT;
	}

    
	void Allocate_Memory(const LOCATION loc);

	COO_Matrix(const int rows, const int cols, const int nz, const CPU_EXISTENCE cpu_exists, const GPU_EXISTENCE gpu_exists);
		
	~COO_Matrix();

	void CopyMatrix_cpu_to_gpu();

	void CopyMatrix_gpu_to_cpu();

	void Deallocate_Memory(const LOCATION loc);
	
   
    //! Copy constructor for COO Matrix class
    /*! 
      A deleted constructor
    */
	COO_Matrix(const COO_Matrix& mat) = delete;

   
    //! Move constructor for COO Matrix class
    /*! 
      A deleted constructor
    */
	COO_Matrix(COO_Matrix&& mat) = delete;

  
   //! Copy assignment operator for COO matrix class
    /*!
        A deleted operator.
    */ 
	COO_Matrix& operator= (const COO_Matrix& mat) = delete;


   //! Move assignment operator for COO matrix class
    /*!
        A deleted operator.
    */ 
	COO_Matrix& operator= (COO_Matrix&& mat) = delete;

};
