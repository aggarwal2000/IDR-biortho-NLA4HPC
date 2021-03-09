/*!
  \file main.cu
  \brief Driver or the entry point of the program 
*/

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include<string> 

#include <cmath>
#include <cassert>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#include "mmio.h"
#include "double_complex.h"
#include "Matrix.h"
#include "Preconditioner.h"
#include "Solver.h"
#include "Read_Write_Data.h"


//! main function(the driver)
/*!
  Takes in command line arguments - matrix A and b files(b is optional) and produces final solution(after solving Ax = b)
*/
int main(int argc, char* argv[])
{
    CSR_Matrix* A = nullptr;
    Dense_Matrix* b = nullptr;

    Read_Matrix_A_and_vector_b(&A, &b, argc, argv); //reads A and b into cpu memory

   

    A->Allocate_Memory(LOCATION::GPU);
    b->Allocate_Memory(LOCATION::GPU);
    A->CopyMatrix_cpu_to_gpu();
    b->CopyMatrix_cpu_to_gpu();
    //A and b on cpu and gpu - ready 

    

    Dense_Matrix x(b->GetRows(), 1, b->GetRows(), ORDER::COLUMN_MAJOR,
     CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::EXISTENT);

    for (int i = 0; i < x.GetRows(); i++) //Set initial guess
    {
        x.GetCPUValues()[i].x = 0;
        x.GetCPUValues()[i].y = 0;
    }

    x.CopyMatrix_cpu_to_gpu(); 



    //Generate preconditioner - Richardson/Jacobi
    Preconditioner* precond = Generate_Preconditioner(PRECONDITIONER_TYPE::RICHARDSON, *A);
    if(precond == nullptr)
    {
    	std::cout << "\nProblem in generating preconditioner.";
    	exit(1);
    }

    Solver solver_obj;
    //Can set atol, rtol, max_iter etc... using solver_obj setter functions
    solver_obj.SetRtol(1e-04);
    solver_obj.SetAtol(1e-20);
   // solver_obj.SetMax_iter(100);
    
    int shadow_space_number = 4;
    solver_obj.PIDR_Solver(*A, *b, x, *precond, shadow_space_number); //Can change value of shadow space number here

    std::cout << "\n\n-----------------------------------Solver info:-----------------------------------------------------------";

   
    std::cout << std::endl << "info: " << solver_obj.GetInfo(); 
    std::cout << std::endl << "iter_resdidual:" << solver_obj.GetIter_residual();
    std::cout << std::endl << "init_residual:" << solver_obj.GetInit_residual();
    std::cout << std::endl << "final_residual:" << solver_obj.GetFinal_residual();
    std::cout << std::endl << "runtime in milliseconds:" << solver_obj.GetRuntimeMilliseconds();
    std::cout << std::endl << "max_iter: " << solver_obj.GetMax_iter();
    std::cout << std::endl << "spmv_count:" << solver_obj.GetSpmv_count();
    std::cout << std::endl << "num_iter:" << solver_obj.GetNum_iter();
    std::cout << std::endl << "full_cycle:" << solver_obj.GetFull_cycle();
    std::cout << std::endl << "rtol:" << solver_obj.GetRtol();
    std::cout << std::endl << "atol:" << solver_obj.GetAtol();

    
    
    if(solver_obj.GetInfo() == "SUCCESS")
    {
        std::cout << "\nSolution is written into file:" << " Vector_x.mtx";
        std::cout << "\nResidual vector is written into file:" << " Vector_res.mtx";

        Write_matrix("Vector_x.mtx" , x);
        Write_matrix("Vector_res.mtx", solver_obj.GetResvec());
    }
   
   
    std::cout << "\nGenerating a log.txt file containing the iteratively computed residuals along with the timings at which they are computed." << std::endl;
    //Generates log
    FILE* fp;
    fp = fopen("log.txt" , "w");
    for(int i=0;i<solver_obj.GetResvec().size();i++)
    {   
        fprintf(fp,"\n Normr = %lg  Timing = %lg",solver_obj.GetResvec()[i].normr,solver_obj.GetResvec()[i].timing);
    }

    
//-----------------------------------------------------------------------------------------------------------
    delete A;
    delete b;
    delete precond;
}


