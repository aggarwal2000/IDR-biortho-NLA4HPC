/*!
 \file Read_Write_Data.h
 \brief Declarations of functions which read data from a file and write into a file
*/

# pragma once

#include<utility>
#include<vector>

#include<cuComplex.h>

typedef cuDoubleComplex DoubleComplex;
class CSR_Matrix;
class Dense_Matrix;
class COO_Matrix;
struct resInfo;

void Read_Matrix_A_and_vector_b(CSR_Matrix** ptr_to_A, Dense_Matrix** ptr_to_b, int argc, char* argv[]);
void Write_matrix(const char *file ,const std::vector<resInfo> &vec);
void Write_matrix(const char *file ,const Dense_Matrix & mat);
