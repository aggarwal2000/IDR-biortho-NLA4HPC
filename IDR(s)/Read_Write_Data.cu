/*!
 \file Read_Write_Data.cu
 \brief Implementation of read, write data functions and their helpers 
*/

#include <algorithm>
#include <iostream>
#include<stdio.h>
#include <vector>
#include <utility>
#include<cassert>
#include <cmath>

#include "double_complex.h"
#include "Matrix.h"
#include "Solver.h"
#include "Read_Write_Data.h"
#include "mmio.h"


//! Writes dense matrix into the specified file
/*!
     Generates a matrix market file with a given name and writes a dense matrix into it.
\param[in] file file name
\param[in] mat reference to dense matrix object which is to be written into the file
*/
void Write_matrix(const char *file ,const Dense_Matrix & mat) 
{
    FILE* fp;
    fp = fopen(file , "w");

    MM_typecode matcode;
    mm_initialize_typecode(&matcode);
    mm_set_array(&matcode);
    mm_set_complex(&matcode);
    mm_set_general(&matcode);
    mm_set_matrix(&matcode);
    
    mm_write_banner(fp,matcode);
   
    int M = mat.GetRows();
    int N = mat.GetCols();
    mm_write_mtx_array_size(fp,M,N);

    
    for(int j=0;j< N;j++)
  {
    for(int i=0;i<M;i++)
    {
        fprintf(fp,"\n%lg %lg",mat.GetSpecificLocationPtrCPU(i, j)->x,mat.GetSpecificLocationPtrCPU(i, j)->y);
    }
  }


  if(fp != stdin)
   fclose(fp);
    
}

 
//! Writes iteratively computed residuals into the specified file
/*! 
    Generates a matirx market file with a given name and writes vector of residuals into it.
\param[in] file file name
\param [in] mat reference to vector containing the residuals and the timings 
*/
void Write_matrix(const char *file , const std::vector<resInfo> &vec)
{
    FILE* fp;
    fp = fopen(file , "w");

    MM_typecode matcode;
    mm_initialize_typecode(&matcode); 
    mm_set_array(&matcode);
    mm_set_real(&matcode);
    mm_set_general(&matcode);
    mm_set_matrix(&matcode);
    
     mm_write_banner(fp,matcode);
 
    int M = vec.size();
    int N = 1 ;
    mm_write_mtx_array_size(fp,M,N);

    for(int i=0;i<M;i++)
    {
        fprintf(fp,"\n%lg",vec[i].normr);
    }
  

  if(fp != stdin)
   fclose(fp);
    
}


//! Used to comapre pair<int,DoubleComplex> objects
/*!
\param[in] a first pair
\param[in] b second pair
\return true if first value of second pair is greater than the first value of the first one
*/
static bool compare_first(
    const std::pair< int, DoubleComplex >& a,
    const std::pair< int, DoubleComplex >& b)
{
    return (a.first < b.first);
}



//! Converts COO Matrix to CSR Matrix
/*!
  \param[in] coo_mat_A refrence to COO Matrix object
  \param[out] A reference to CSR Matrix object
 */
void Convert_COO_to_CSR(const COO_Matrix& coo_mat_A, CSR_Matrix& A)
{
    //std::cout << "Welcome";

    std::vector< std::pair< int, DoubleComplex > > rowval;

    int nz = coo_mat_A.Getnz();
    int M = coo_mat_A.GetRows();

    /* convert the COO matrix to CSR */


    int* row_ptr_A = A.GetCPURowPtr();
    int* col_ind_A = A.GetCPUColInd();
    DoubleComplex* val_A = A.GetCPUValues();
    int* I = coo_mat_A.GetCPURowInd();
    int* J = coo_mat_A.GetCPUColInd();
    DoubleComplex* valt = coo_mat_A.GetCPUValues();

    // original code from  Nathan Bell and Michael Garland
    for (int i = 0; i < M; i++)
        row_ptr_A[i] = 0;

    for (int i = 0; i < nz; i++)
        row_ptr_A[I[i]]++;

    // cumulative sum the nnz per row to get row[]
    int cumsum;
    cumsum = 0;
    for (int i = 0; i < M; i++) {
        int temp = row_ptr_A[i];
        row_ptr_A[i] = cumsum;
        cumsum += temp;
    }
    row_ptr_A[M] = nz;

    // write Aj,Ax into Bj,Bx
    for (int i = 0; i < nz; i++) {
        int row_ = I[i];
        int dest = row_ptr_A[row_];
        col_ind_A[dest] = J[i];
        val_A[dest] = valt[i];
        row_ptr_A[row_]++;
    }

    int last;
    last = 0;
    for (int i = 0; i <= M; i++) {
        int temp = (row_ptr_A)[i];
        (row_ptr_A)[i] = last;
        last = temp;
    }

    (row_ptr_A)[M] = nz;

    // sort column indices within each row
    // copy into vector of pairs (column index, value), sort by column index
    for (int k = 0; k < M; ++k) {
        int kk = (row_ptr_A)[k];
        int len = (row_ptr_A)[k + 1] - row_ptr_A[k];
        rowval.resize(len);
        for (int i = 0; i < len; ++i) {
            rowval[i] = std::make_pair(col_ind_A[kk + i], val_A[kk + i]);
        }
        std::sort(rowval.begin(), rowval.end(), compare_first);
        for (int i = 0; i < len; ++i) {
            col_ind_A[kk + i] = rowval[i].first;
            val_A[kk + i] = rowval[i].second;
        }
    }



}



//! Reads elements from a file and fills in a symmetric COO matrix
/*!
 \param[in,out] f_A file pointer
 \param[in,out] coo_mat_A reference to COO matrix object
 \param[in] nz_read_from_file number of non zero elements read from given file
 \param[in] matcode_A matrix banner read from given file
*/
void Fill_symmetric_COO(FILE* f_A , COO_Matrix& coo_mat_A ,const int nz_read_from_file ,const MM_typecode matcode_A)
{
    int I;
    int J;
    double Re, Im;
    //read matrix A (in COO format)

    if (mm_is_complex(matcode_A))
    {    

        int other_i = 0;
        for (int i = 0; i < nz_read_from_file; i++)
        {

            fscanf(f_A, "%d %d %lg %lg\n", &I, &J, &Re, &Im);
            I--;  // adjust from 1-based to 0-based
            J--;
            (coo_mat_A.GetCPURowInd())[other_i] = I;
            (coo_mat_A.GetCPUColInd())[other_i] = J;
            (coo_mat_A.GetCPUValues())[other_i].x = Re;
            (coo_mat_A.GetCPUValues())[other_i].y = Im;
             other_i++;

            if(I != J)
            {
               (coo_mat_A.GetCPURowInd())[other_i] = J;
               (coo_mat_A.GetCPUColInd())[other_i] = I;
               (coo_mat_A.GetCPUValues())[other_i].x = Re;
               (coo_mat_A.GetCPUValues())[other_i].y = Im;
               other_i++;
            }
        }

        coo_mat_A.Set_nz(other_i);
    }
    else if (mm_is_real(matcode_A) || mm_is_integer(matcode_A))
    {   
        int other_i = 0;
        for (int i = 0; i < nz_read_from_file; i++)
        {
            fscanf(f_A, "%d %d %lg \n", &I, &J, &Re);
            I--;   // adjust from 1-based to 0-based 
            J--;
            Im = 0;
            (coo_mat_A.GetCPURowInd())[other_i] = I;
            (coo_mat_A.GetCPUColInd())[other_i] = J;
            (coo_mat_A.GetCPUValues())[other_i].x = Re;
            (coo_mat_A.GetCPUValues())[other_i].y = Im;
             other_i++;

            if(I != J)
            {
               (coo_mat_A.GetCPURowInd())[other_i] = J;
               (coo_mat_A.GetCPUColInd())[other_i] = I;
               (coo_mat_A.GetCPUValues())[other_i].x = Re;
               (coo_mat_A.GetCPUValues())[other_i].y = Im;
               other_i++;
            }
        }

         coo_mat_A.Set_nz(other_i);
    }
    else
    {
        printf("This case is not handled!");
        exit(1);
    }
   
}


//! Reads elements from a file and fills in a hermitian COO matrix
/*!
 \param[in,out] f_A file pointer
 \param[in,out] coo_mat_A reference to COO matrix object
 \param[in] nz_read_from_file number of non zero elements read from given file
 \param[in] matcode_A matrix banner read from given file
*/
void Fill_hermitian_COO( FILE* f_A , COO_Matrix& coo_mat_A , const int nz_read_from_file,const MM_typecode matcode_A)
{
     int I;
    int J;
    double Re, Im;
    //read matrix A (in COO format)

    if (mm_is_complex(matcode_A))
    {    

        int other_i = 0;
        for (int i = 0; i < nz_read_from_file; i++)
        {

            fscanf(f_A, "%d %d %lg %lg\n", &I, &J, &Re, &Im);
            I--;  // adjust from 1-based to 0-based
            J--;
            (coo_mat_A.GetCPURowInd())[other_i] = I;
            (coo_mat_A.GetCPUColInd())[other_i] = J;
            (coo_mat_A.GetCPUValues())[other_i].x = Re;
            (coo_mat_A.GetCPUValues())[other_i].y = Im;
             other_i++;

            if(I != J)
            {
               (coo_mat_A.GetCPURowInd())[other_i] = J;
               (coo_mat_A.GetCPUColInd())[other_i] = I;
               (coo_mat_A.GetCPUValues())[other_i].x = Re;
               (coo_mat_A.GetCPUValues())[other_i].y = -1*Im;
               other_i++;
            }
        }

        coo_mat_A.Set_nz(other_i);
    }
    else if (mm_is_real(matcode_A) || mm_is_integer(matcode_A))
    {   
        int other_i = 0;
        for (int i = 0; i < nz_read_from_file; i++)
        {
            fscanf(f_A, "%d %d %lg \n", &I, &J, &Re);
            I--;   // adjust from 1-based to 0-based 
            J--;
            Im = 0;
            (coo_mat_A.GetCPURowInd())[other_i] = I;
            (coo_mat_A.GetCPUColInd())[other_i] = J;
            (coo_mat_A.GetCPUValues())[other_i].x = Re;
            (coo_mat_A.GetCPUValues())[other_i].y = Im;
             other_i++;

            if(I != J)
            {
               (coo_mat_A.GetCPURowInd())[other_i] = J;
               (coo_mat_A.GetCPUColInd())[other_i] = I;
               (coo_mat_A.GetCPUValues())[other_i].x = Re;
               (coo_mat_A.GetCPUValues())[other_i].y = -1*Im;
               other_i++;
            }
        }

         coo_mat_A.Set_nz(other_i);
    }
    else
    {
        printf("This case is not handled!");
        exit(1);
    }
}



//! Reads elements from a file and fills in a skew-symmetric COO matrix
/*!
 \param[in,out] f_A file pointer
 \param[in,out] coo_mat_A reference to COO matrix object
 \param[in] nz_read_from_file number of non zero elements read from given file
 \param[in] matcode_A matrix banner read from given file
*/
void Fill_skew_COO(FILE* f_A , COO_Matrix& coo_mat_A , const int nz_read_from_file,const MM_typecode matcode_A)
{
     int I;
    int J;
    double Re, Im;
    //read matrix A (in COO format)

    if (mm_is_complex(matcode_A))
    {    

        int other_i = 0;
        for (int i = 0; i < nz_read_from_file; i++)
        {

            fscanf(f_A, "%d %d %lg %lg\n", &I, &J, &Re, &Im);
            I--;  // adjust from 1-based to 0-based
            J--;
            (coo_mat_A.GetCPURowInd())[other_i] = I;
            (coo_mat_A.GetCPUColInd())[other_i] = J;
            (coo_mat_A.GetCPUValues())[other_i].x = Re;
            (coo_mat_A.GetCPUValues())[other_i].y = Im;
             other_i++;

            if(I != J)
            {
               (coo_mat_A.GetCPURowInd())[other_i] = J;
               (coo_mat_A.GetCPUColInd())[other_i] = I;
               (coo_mat_A.GetCPUValues())[other_i].x = -1*Re;
               (coo_mat_A.GetCPUValues())[other_i].y = -1*Im;
               other_i++;
            }
        }

        coo_mat_A.Set_nz(other_i);
    }
    else if (mm_is_real(matcode_A) || mm_is_integer(matcode_A))
    {   
        int other_i = 0;
        for (int i = 0; i < nz_read_from_file; i++)
        {
            fscanf(f_A, "%d %d %lg \n", &I, &J, &Re);
            I--;   // adjust from 1-based to 0-based 
            J--;
            Im = 0;
            (coo_mat_A.GetCPURowInd())[other_i] = I;
            (coo_mat_A.GetCPUColInd())[other_i] = J;
            (coo_mat_A.GetCPUValues())[other_i].x = Re;
            (coo_mat_A.GetCPUValues())[other_i].y = Im;
             other_i++;

            if(I != J)
            {
               (coo_mat_A.GetCPURowInd())[other_i] = J;
               (coo_mat_A.GetCPUColInd())[other_i] = I;
               (coo_mat_A.GetCPUValues())[other_i].x = -1*Re;
               (coo_mat_A.GetCPUValues())[other_i].y = -1*Im;
               other_i++;
            }
        }

         coo_mat_A.Set_nz(other_i);
    }
    else
    {
        printf("This case is not handled!");
        exit(1);
    }
}


//! Reads elements from a file and fills in a general COO matrix
/*!
 \param[in,out] f_A file pointer
 \param[in,out] coo_mat_A reference to COO matrix object
 \param[in] nz_read_from_file number of non zero elements read from given file
 \param[in] matcode_A matrix banner read from given file
*/
void Fill_general_COO(FILE* f_A , COO_Matrix& coo_mat_A , const int nz_read_from_file , const MM_typecode matcode_A)
{
    int I;
    int J;
    double Re, Im;
    //read matrix A (in COO format)

    if (mm_is_complex(matcode_A))
    {
        for (int i = 0; i < nz_read_from_file; i++)
        {

            fscanf(f_A, "%d %d %lg %lg\n", &I, &J, &Re, &Im);
            I--;  // adjust from 1-based to 0-based
            J--;
            (coo_mat_A.GetCPURowInd())[i] = I;
            (coo_mat_A.GetCPUColInd())[i] = J;
            (coo_mat_A.GetCPUValues())[i].x = Re;
            (coo_mat_A.GetCPUValues())[i].y = Im;
        }
    }
    else if (mm_is_real(matcode_A) || mm_is_integer(matcode_A))
    {
        for (int i = 0; i < nz_read_from_file; i++)
        {
            fscanf(f_A, "%d %d %lg \n", &I, &J, &Re);
            I--;   // adjust from 1-based to 0-based 
            J--;
            Im = 0;
            (coo_mat_A.GetCPURowInd())[i] = I;
            (coo_mat_A.GetCPUColInd())[i] = J;
            (coo_mat_A.GetCPUValues())[i].x = Re;
            (coo_mat_A.GetCPUValues())[i].y = Im;
        }
    }
    else
    {
        printf("This case is not handled!");
        exit(1);
    }
   
}


//! Reads in or generates matrix A, vector b from the given files
/*!

 Reads matrix A from the file passed as command line argument and creates a CSR matrix object
 Reads vector b from the file if the file name is passed as command line argument,
  else generates it on its own and creates a dense matrix object

\param[out] ptr_to_A pointer to CSR Matrix pointer
\param[out] ptr_to_b pointer to dense matrix pointer
\param[in] argc number of command line arguments
\param[in] argv array of strings passed as command line argument
*/ 
void Read_Matrix_A_and_vector_b(CSR_Matrix** ptr_to_A, Dense_Matrix** ptr_to_b, int argc, char* argv[])
{
    int ret_code_A, ret_code_b;
    MM_typecode matcode_A, matcode_b;
    FILE* f_A, * f_b;

    int M, N, nz; //for matrix A
    int rows_b, cols_b; //for vector b

    
    if (argc < 2)
    {
        fprintf(stderr, "  Usage: %s [matrix-market-filename to read matrix A] [matrix market file name to read vector b] \n",
            argv[0]);

        printf("\n OR \n  Usage: %s  [matrix-market-filename to read matrix A] \n ", argv[0]);

        exit(1);
    }
   
    f_A = fopen(argv[1], "r");

    if(f_A == NULL)
      {
         std::cout << "\nIssue with matrix A file";
         exit(1);
      }

    if (mm_read_banner(f_A, &matcode_A) != 0)
    {
        std::cout << "Could not process Matrix Market banner.\n" << std::endl;
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */


    if (!(mm_is_matrix(matcode_A) && mm_is_coordinate(matcode_A)))
    {
        std::cout << "Sorry , this application does not support this type for matrix A!" << std::endl;
        std::cout << "Matrix Market File " << mm_typecode_to_str(matcode_A) << std::endl;
        exit(1);
    }

    /* find out size of sparse matrix A .... */
    ret_code_A = mm_read_mtx_crd_size(f_A, &M, &N, &nz);
    if (ret_code_A != 0)
        exit(1);

    if (M != N)
    {
        printf("\nMatrix is not sqaure; not supported by the application");
        exit(1);
    }

    

   
  if (argc == 3)
  {
        if ((f_b = fopen(argv[2], "r")) == NULL)
        {
            std::cout << " Issue with vector b file\n" << std::endl;
            exit(1);
        }

        if (mm_read_banner(f_b, &matcode_b) != 0)
        {
            std::cout << "Could not process Matrix Market banner.\n" << std::endl;
            exit(1);
        }

        if (!(mm_is_matrix(matcode_b) && mm_is_array(matcode_b)))
        {
            std::cout << "Sorry , this appication does not support this type for vector b !" << std::endl;
            std::cout << "Matrix Market File " << mm_typecode_to_str(matcode_A) << std::endl;
            exit(1);
        }

        // find out size of vector b 
        ret_code_b = mm_read_mtx_array_size(f_b, &rows_b, &cols_b);
        if (ret_code_b != 0)
            exit(1);


        if ((cols_b != 1) || (rows_b != M))
        {
            std::cout << "Sorry , System Ax = b makes no sense !!\n";
            exit(1);
        }

        // M = N = rows_b

        *ptr_to_b = new Dense_Matrix(M, 1, M, ORDER::COLUMN_MAJOR, CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::NON_EXISTENT);

        Dense_Matrix* b = *ptr_to_b;
        //read vector b
        double re;
        double im;
        for (int i = 0; i < M; i++)
        {
            if (mm_is_complex(matcode_b))
            {
                fscanf(f_b, "%lg %lg\n", &re, &im);
                (b->GetCPUValues())[i].x = re;
                (b->GetCPUValues())[i].y = im;

            }
            else if (mm_is_integer(matcode_b) || mm_is_real(matcode_b))
            {
                fscanf(f_b, "%lg \n", &re);
                (b->GetCPUValues())[i].x = re;
                (b->GetCPUValues())[i].y = 0;
            }
            else
            {
                printf(" This case is not handled ...");
                exit(1);
            }
        }

        if (f_b != stdin) fclose(f_b);

     }
    else if (argc == 2)
    {
        *ptr_to_b = new Dense_Matrix(N, 1, N, ORDER::COLUMN_MAJOR, CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::NON_EXISTENT);

        Dense_Matrix* b = *ptr_to_b;
        //read vector b

        for (int i = 0; i < M; i++)
        {
            (b->GetCPUValues())[i].x = 1;
            (b->GetCPUValues())[i].y = 0;
        }
        
    }

    
    int temp_nz = nz;

   
    if(mm_is_general(matcode_A))
    {
        temp_nz = nz;

    }
    else if(mm_is_hermitian(matcode_A) || mm_is_symmetric(matcode_A) || mm_is_skew(matcode_A))
    {
        temp_nz = 2*nz;
    }
    else
    {
         printf("This case is not handled yet..");
         exit(1);
    }
    
    //reserve memory for matrix A - in COO format 
    COO_Matrix coo_mat_A(M, N, temp_nz, CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::NON_EXISTENT); //Use Set_nz() afterwards...
    
    if(mm_is_general(matcode_A))
        Fill_general_COO(f_A , coo_mat_A, nz , matcode_A);
    else if(mm_is_symmetric(matcode_A))
        Fill_symmetric_COO(f_A , coo_mat_A, nz , matcode_A);
    else if(mm_is_hermitian(matcode_A))
        Fill_hermitian_COO(f_A , coo_mat_A , nz , matcode_A);
    else if(mm_is_skew(matcode_A))
        Fill_skew_COO(f_A , coo_mat_A ,nz , matcode_A);
    else
    {
         printf("This case is not handled yet..");
         exit(1);
    }

    

    *ptr_to_A = new CSR_Matrix(coo_mat_A.GetRows(), coo_mat_A.GetCols(), coo_mat_A.Getnz(), CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::NON_EXISTENT);  //A is a genral N x N complex matrix
    CSR_Matrix* A = *ptr_to_A;

    if (f_A != stdin) fclose(f_A);

    Convert_COO_to_CSR(coo_mat_A, *A);



}











