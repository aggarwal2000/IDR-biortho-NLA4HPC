/*!
\file location_enums.h
\brief Contains defintion of common enum classes used in various preconditioner and matrix classes
*/
# pragma once


//! enum class which defines location for memory allocation
/*!
  This enum class is used to indicate the location where the memory is to be allocated to 
  store internals(large arrays) of matrix and preconditioner class objects.
*/
enum class LOCATION {

	CPU, /*!< indicates intent to allocate memory on CPU for storage of matrix/preconditioner object internals*/
	GPU  /*!< indicates intent to allocate memory on GPU for storage of matrix/preconditioner object internals*/
};




//! enum class which defines presence/absence of matrix and preconditioner class objects internals(large arrays) in CPU(host) memory
/*!
 For example, in case of CSR matrix, value of enum class variable = CPU_EXISTENCE::EXISTENT would mean that
 array of values, column indices and row pointers (some of the attributes of CSR Matrix class) is located in the CPU memory. 
*/ 
enum class CPU_EXISTENCE {
	EXISTENT, /*!<  means internals(large arrays) do exist at the dedicated location*/
	NON_EXISTENT  /*!<  means internals(large arrays) do not exist at the dedicated location*/
};



//! enum class which defines presence/absence of matrix and preconditioner class objects internals(large arrays) in GPU(device) memory 
/*!
 For example, in case of CSR matrix, value of enum class variable = GPU_EXISTENCE::EXISTENT would mean that
 array of values, column indices and row pointers (some of the attributes of CSR Matrix class) is located in the GPU memory. 
*/ 
enum class GPU_EXISTENCE {
	EXISTENT, /*!< means internals(large arrays) do exist at the dedicated location*/
	NON_EXISTENT /*!< means internals(large arrays) do not exist at the dedicated location*/
};


