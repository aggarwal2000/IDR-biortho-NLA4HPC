/*!
 \file PIDR_Solver.cu
 \brief Implementation of Solver::PIDR_Solver() and its helper functions 
*/

#include<iostream>
#include<stdio.h>
 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "double_complex.h"
#include "Matrix.h"
#include "Solver.h"
#include "kernels.h"
#include "Preconditioner.h"
#include "header.h"



//! Helper function used to perform initialization for the PIDR solver
void PIDR_Initialization(Dense_Matrix& x, Dense_Matrix& P, Dense_Matrix& U, Dense_Matrix& G, Dense_Matrix& M);



void PIDR_Initialization(Dense_Matrix& x ,Dense_Matrix& P, Dense_Matrix& U, Dense_Matrix& G, Dense_Matrix& M)
{
    const int N = U.GetRows();
    const int s = M.GetRows();
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid1(ceil(static_cast<double>(N) / static_cast<double>(THREADS_PER_BLOCK)));
    dim3 grid2(ceil(static_cast<double>(s) / static_cast<double>(THREADS_PER_BLOCK)));

    
    
    //Initialize soultion vector x in case it is not allocated on CPU memory
    //It is assumed to be initialized if it is present on the CPU 
    if(x.ExistsCPU() == false && x.ExistsGPU() == false)
    {
            x.Allocate_Memory(LOCATION::CPU);
            x.Allocate_Memory(LOCATION::GPU);
            Fill_in_Zeroes_Dense_Matrix << <grid1, block >> > (x.GetRows(), x.GetCols(), x.GetLda(), x.GetGPUValues());
            x.CopyMatrix_gpu_to_cpu();
    }
   else if(x.ExistsGPU() == false ) //assuming it was initialized in main(), as it present on CPU memory
    {
         x.Allocate_Memory(LOCATION::GPU);
         x.CopyMatrix_cpu_to_gpu();         
    }   
   else if(x.ExistsCPU() == false ) //so in main(), it was not initialized
    {
       x.Allocate_Memory(LOCATION::CPU);
       Fill_in_Zeroes_Dense_Matrix << <grid1, block >> > (x.GetRows(), x.GetCols(), x.GetLda(), x.GetGPUValues());
       x.CopyMatrix_gpu_to_cpu();

    }
    else //assuming it was initialized in main(), as it is present on CPU 
    {
        x.CopyMatrix_cpu_to_gpu();        
    }

    //Initializing P
    if (P.ExistsCPU() == false)
        P.Allocate_Memory(LOCATION::CPU);
    Fill_in_random_numbers_Dense_Matrix(P.GetRows(), P.GetCols(), P.GetLda(), P.GetCPUValues());
    //Now write code to orthogonalize it
    //Gram_Schmidt_cpu(P);
     if (P.ExistsGPU() == false)
        P.Allocate_Memory(LOCATION::GPU);
    P.CopyMatrix_cpu_to_gpu();
    Gram_Schmidt(P);
    P.Deallocate_Memory(LOCATION::CPU);

    //Initilaizing U
    if(U.ExistsGPU() == false)
        U.Allocate_Memory(LOCATION::GPU);
    Fill_in_Zeroes_Dense_Matrix << <grid1, block >> > (U.GetRows(), U.GetCols(), U.GetLda(), U.GetGPUValues());

    //Initializing G
    if(G.ExistsGPU() == false)
        G.Allocate_Memory(LOCATION::GPU);
    Fill_in_Zeroes_Dense_Matrix << <grid1, block >> > (G.GetRows(), G.GetCols(), G.GetLda(), G.GetGPUValues());

    //Initializing M
     if(M.ExistsGPU() == false)
        M.Allocate_Memory(LOCATION::GPU);
    Fill_in_identity_Dense_Matrix << <grid2, block >> > (M.GetRows(), M.GetLda(), M.GetGPUValues());
    if(M.ExistsCPU() == false)
        M.Allocate_Memory(LOCATION::CPU);
    M.CopyMatrix_gpu_to_cpu();
}


//! Preconditioned Induced Dimension Reduction solver
/*!
    Purpose is to solve a system of linear equations Ax = b
    \param[in] A reference to CSR matrix object(input matrix A)
    \param[in] b reference to Dense matrix object(RHS b)
    \param[in,out] x reference to Dense matrix object (solution approximation x)
    \param[in] precond reference to a preconditioner object
    \param[in] s shadow space number 
*/
void Solver::PIDR_Solver(const CSR_Matrix& A, const Dense_Matrix& b, Dense_Matrix& x,const Preconditioner& precond, const int s) 
{
    this->name = SOLVER_NAME::PIDR;
    //set max iterations ,atol, rtol if not set by the user
    if(this->max_iter == 0) //means not initialized by user
      this->max_iter = 2 * A.GetRows();
    if(this->atol == 0) //means not initialized by user
        this->atol = pow(10,-12);
    if(this->rtol == 0) //means not initialized by user
        this->rtol = pow(10,-4);

    //other solver parameters initialized when the solver object was created


    //internal user parameters
    const int smoothing_operation = 1;
    const double angle = 0.7;

    //Check if A and b are on gpu
    if (A.ExistsGPU() == false || b.ExistsGPU() == false) 
    {
        this->info = SOLVER_INFO::UNALLOCATED_INPUT_MATRICES;
        return;
    }

    if (A.GetRows() != A.GetCols())
    {
        this->info = SOLVER_INFO::ERR_NOT_SUPPORTED;  //application only supports square matrices
        return;
    }

    

   //allocate matrices and vectors
    int lda_P_U_G;
    int lda_M;
    if (s > 1)
    {
        lda_P_U_G = Roundup(A.GetRows(), 32);
        lda_M = Roundup(s, 32);
    }
    else
    {
        lda_P_U_G = A.GetRows();
        lda_M = s;
    }
    Dense_Matrix P(A.GetRows(), s, lda_P_U_G, ORDER::COLUMN_MAJOR, CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::EXISTENT);
    Dense_Matrix U(A.GetRows(), s, lda_P_U_G, ORDER::COLUMN_MAJOR, CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::EXISTENT);
    Dense_Matrix G(A.GetRows(), s, lda_P_U_G, ORDER::COLUMN_MAJOR, CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::EXISTENT);
    Dense_Matrix M(s, s, lda_M, ORDER::COLUMN_MAJOR, CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::EXISTENT);
    Dense_Matrix r(A.GetRows(), 1, A.GetRows(), ORDER::COLUMN_MAJOR, CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::EXISTENT);
    Dense_Matrix f(s, 1, s, ORDER::COLUMN_MAJOR, CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::EXISTENT);
    Dense_Matrix c(s, 1, s, ORDER::COLUMN_MAJOR, CPU_EXISTENCE::EXISTENT, GPU_EXISTENCE::EXISTENT);
    Dense_Matrix v(A.GetRows(), 1, A.GetRows(), ORDER::COLUMN_MAJOR, CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::EXISTENT);
    Dense_Matrix t(A.GetRows(), 1, A.GetRows(), ORDER::COLUMN_MAJOR, CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::EXISTENT);
    //allocated M , f ,c --> on host as well

    double abs_rho;
    DoubleComplex omega = { 1,0 };
    DoubleComplex rho;
    int innerflag;

    DoubleComplex alpha;
    DoubleComplex beta;
    double residual;
    double normr;
    double normb;
    double relres;

   // ||b||
    normb = Compute_L2Norm(b, 0, 0, b.GetRows() - 1);


    if (normb == 0.0)
    {
        info = SOLVER_INFO::SUCCESS;
        Scaling_Vector(x, 0, 0, x.GetRows() - 1, { 0,0 });
        x.CopyMatrix_gpu_to_cpu();
        this->init_residual = 0.0;
        this->final_residual = this->init_residual;
        this->iter_residual = this->init_residual;
        return;
    }

    this->num_iter = 0;

    // r = b - Ax 
    Compute_Residual(A, b, x, r);

    // ||r||
    normr = Compute_L2Norm(r, 0, 0, r.GetRows() - 1);
    this->init_residual = normr;
    this->final_residual = this->init_residual;
    this->iter_residual = this->init_residual;

    relres = normr / normb;

   // std::cout << "\nresidualNorm: " << normr << "  normb: " << normb << " in beginning";

   //  std::cout << "\nRel residualNorm:" << relres << " in beginning";
   // std::cout << "\nTiming:" << 0.0;
    this->resvec.push_back({normr , 0.0});

    if (relres <= this->rtol || normr < this->atol) //Check if initial guess is good enough
    {
        std::cout << "\nInitial guess is good enough\n";
        this->info = SOLVER_INFO::SUCCESS;
        x.CopyMatrix_gpu_to_cpu();
        return;
    }


    PIDR_Initialization(x,P, U, G, M);

    //useful for smoothing operation
    Dense_Matrix xs(x.GetRows(), x.GetCols(), x.GetLda(), x.GetOrder(), CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::NON_EXISTENT);
    Dense_Matrix rs(r.GetRows(), r.GetCols(), r.GetLda(), r.GetOrder(), CPU_EXISTENCE::NON_EXISTENT, GPU_EXISTENCE::NON_EXISTENT);
    DoubleComplex gamma;

    if (smoothing_operation == 1)
    {
        xs = x; //copy all cpu_gpu values
        rs = r;
    }

    innerflag = 0;
    M.CopyMatrix_gpu_to_cpu();
    //-------------------------------------------------------------start time--------------------------------------------
    cudaEvent_t start, stop; //chronometry
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    cudaEvent_t tempo2;
    cudaEventCreate(&tempo2);

    cudaEventRecord(start);
    

    while (this->num_iter < this->max_iter)
    {
        //f = P' * r (here ' is conjugate transpose)
        Compute_HermitianMatrix_vec_mul(P, 0, P.GetCols() - 1, 0, P.GetRows() - 1, r, 0, 0, r.GetRows() - 1, f, 0, 0, f.GetRows() - 1);

        //shadow space loop
        for (int k = 0; k < s; k++)
        {

           //c(k:s-1) = M(k:s-1,k:s-1)\f(k:s-1)
            // M.CopyMatrix_gpu_to_cpu(); //No need to copy full matrix...
            M.CopyMatrix_gpu_to_cpu(k,s-1,k,s-1);
            f.CopyMatrix_gpu_to_cpu(0, 0, k, s - 1);
            c.CopyMatrix_gpu_to_cpu(0, 0, k, s - 1); 
            Triangular_Solve(M, k, s - 1, k, s - 1, c, 0, k, s - 1, f, 0, k, s - 1);
            c.CopyMatrix_cpu_to_gpu(0, 0, k, s - 1);


            //#### v = r - G(:,k:s-1) c(k:s-1)
            // v =  G(:,k:s-1) c(k:s-1)
            Compute_GeMV(G, k, s - 1, 0, G.GetRows() - 1, c, 0, k, s - 1, v, 0, 0, v.GetRows() - 1);
            // v = -v + r
            Compute_Vector_Linear_Combination({ -1,0 }, v, 0, 0, v.GetRows() - 1, { 1,0 }, r, 0, 0, r.GetRows() - 1, v, 0, 0, v.GetRows() - 1);


            //v = preconditioner*v
            precond.ApplyPreconditioner(v, v);



            //#### U(:,k) = omega * v + U(:,k:s-1) c(k:s-1)
            //U(:, k) =  U(:, k : s-1) c(k:s-1)
            Compute_GeMV(U, k, s - 1, 0, U.GetRows() - 1, c, 0, k, s - 1, U, k, 0, U.GetRows() - 1);
            //U(:,k) =  U(:,k) + omega*v
             Perform_axpy( omega, v, 0, 0, v.GetRows() - 1, U, k, 0, U.GetRows() - 1);



            // G(:,k) = A U(:,k)
            Compute_CSR_SpMv(A, U, k, 0, U.GetRows() - 1, G, k, 0, G.GetRows() - 1);
            this->spmv_count = this->spmv_count + 1;


           

            //bi-orthogonalize new basis vectors
            for (int i = 0; i < k; i++)
            {
               // if (M.GetSpecificLocationPtrCPU(i, i)->x == 0 && M.GetSpecificLocationPtrCPU(i, i)->y == 0)
                   // printf("\nZero Dr i,i");

               // M(i,i,i,i) for 0<=i<k on cpu is already updated with GPU value
                
                // alpha = P(:,i)' G(:,k) / M(i,i)
                alpha = Compute_Inner_Product(G, k, 0, G.GetRows() - 1, P, i, 0, P.GetRows() - 1) / *(M.GetSpecificLocationPtrCPU(i, i));
                //Inner Product does:   <u,v> = u transpose * v conjugate ; so call fn accordingly
                //return dot product of 2 vectors... // <v,u> = u hermitian * v

                // G(:,k) = G(:,k) - alpha * G(:,i)
                 Perform_axpy( -1 * alpha, G, i, 0, G.GetRows() - 1, G, k, 0, G.GetRows() - 1);
                // U(:,k) = U(:,k) - alpha* U(:,i)
                 Perform_axpy( -1 * alpha, U, i, 0, U.GetRows() - 1, U, k, 0, U.GetRows() - 1);

            }




            //  M(k:s-1, k) = P(:, k : s-1)' G(:,k)
            Compute_HermitianMatrix_vec_mul(P, k, s - 1, 0, P.GetRows() - 1, G, k, 0, G.GetRows() - 1, M, k, k, s - 1);

            M.CopyMatrix_gpu_to_cpu(k,k,k,k); 

            //Check M(k,k) == 0
            if (M.GetSpecificLocationPtrCPU(k, k)->x == 0 && M.GetSpecificLocationPtrCPU(k, k)->y == 0)
            {
               // printf("\nZero Dr k,k");
                this->info = SOLVER_INFO::DIVERGENCE;
                innerflag = 1;
                break;
            }
            
            //f(k) on CPU is already updated with value on GPU
            //beta = f(k)/M(k,k)
            beta = *f.GetSpecificLocationPtrCPU(k, 0) / *M.GetSpecificLocationPtrCPU(k, k);

            //check for nan
            if (Is_Finite(beta) == false)
            {
               // printf("Nan / inf true");
                innerflag = 1;
                this->info = SOLVER_INFO::DIVERGENCE;
                break;
            }

            // r = r - beta * G(:,k)
             Perform_axpy( -1 * beta, G, k, 0, G.GetRows() - 1, r, 0, 0, r.GetRows() - 1);



            // x = x + beta * U(:,k)
             Perform_axpy( beta, U, k, 0, U.GetRows() - 1, x, 0, 0, x.GetRows() - 1);


            if (smoothing_operation == 1)
            {
                //t = rs - r
                Compute_Vector_Linear_Combination({ 1,0 }, rs, 0, 0, rs.GetRows() - 1, { -1,0 }, r, 0, 0, r.GetRows() - 1, t, 0, 0, t.GetRows() - 1);

                //gamma = t'rs / t't
                gamma = Compute_Inner_Product(rs, 0, 0, rs.GetRows() - 1, t, 0, 0, t.GetRows() - 1) / Compute_Inner_Product(t, 0, 0, t.GetRows() - 1, t, 0, 0, t.GetRows() - 1);

                DoubleComplex temp_gamma = make_cuDoubleComplex(1, 0) - gamma;

                //rs = rs - gamma*(rs - r)
                Compute_Vector_Linear_Combination(temp_gamma, rs, 0, 0, rs.GetRows() - 1, gamma, r, 0, 0, r.GetRows() - 1, rs, 0, 0, rs.GetRows() - 1);

                //xs = xs - gamma*(xs - x)
                Compute_Vector_Linear_Combination(temp_gamma, xs, 0, 0, xs.GetRows() - 1, gamma, x, 0, 0, x.GetRows() - 1, xs, 0, 0, xs.GetRows() - 1);

                //normr = ||rs||
                normr = Compute_L2Norm(rs, 0, 0, rs.GetRows() - 1);
            }
            else
            {  
                //normr = ||r||
                normr = Compute_L2Norm(r, 0, 0, r.GetRows() - 1);
            }



            relres = normr / normb;

           // std::cout << "\nRel Residual Norm is:" << relres << " after iteration : " << this->num_iter + 1;

            cudaEventRecord(tempo2);
            cudaEventSynchronize(tempo2);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, tempo2);
           // std::cout << "\nTiming:" << milliseconds;
            this->resvec.push_back({ normr , milliseconds }); //store normr and timings

            this->num_iter++;

            if (relres <= this->rtol || normr < this->atol) //check convergence
            {
                innerflag = 2;
                this->info = SOLVER_INFO::SUCCESS;
                break;
            }
            if (this->num_iter >= this->max_iter) //reached iteration limit
            {
                innerflag = 3;
                break;
            }

    
            if (k + 1 < s) //non-last s iteration
            { 
                 // f(k+1:s-1) = f(k+1:s-1) - beta*M(k+1:s-1,k)
                Scaling_Vector(f, 0, 0, k, { 0,0 });
                Perform_axpy( -1 * beta, M, k, k + 1, s - 1, f, 0, k + 1, s - 1);
            }

        } //end for

        //check convergence(inner_flag : 2)  or iteration limit(inner_flag:3)  or invalid result of inner loop(inner_flag:1)
        if (innerflag > 0)
            break;

        // v = preconditioner*r
        precond.ApplyPreconditioner(r, v);

        // t = Av
        Compute_CSR_SpMv(A, v, 0, 0, v.GetRows() - 1, t, 0, 0, t.GetRows() - 1);
        this->spmv_count++;

        DoubleComplex r_t = Compute_Inner_Product(r, 0, 0, r.GetRows() - 1, t, 0, 0, t.GetRows() - 1);
        DoubleComplex t_t =  Compute_Inner_Product(t, 0, 0, t.GetRows() - 1, t, 0, 0, t.GetRows() - 1);
        double normt = sqrt(t_t.x);
        omega = r_t / t_t; //omega = t'r / t't
        rho = r_t / (normt * Compute_L2Norm(r, 0, 0, r.GetRows() - 1)); //rho = t'r / ||t||*||r|| 
        abs_rho = fabs(rho);
        if (abs_rho < angle) //Calculation of omega using maintaing the convergnece strategy
        {
            omega = omega * (angle / abs_rho);
        }

        if (omega.x == 0 && omega.y == 0)
        {
            this->info = SOLVER_INFO::DIVERGENCE;
            break;
        }

        // r = r - omega*t
         Perform_axpy( -1 * omega, t, 0, 0, t.GetRows() - 1, r, 0, 0, r.GetRows() - 1);

        // x = x + omega*v
         Perform_axpy( omega, v, 0, 0, v.GetRows() - 1, x, 0, 0, x.GetRows() - 1); 

        if (smoothing_operation == 1)
        {
            //t = rs - r
            Compute_Vector_Linear_Combination({ 1,0 }, rs, 0, 0, rs.GetRows() - 1, { -1,0 }, r, 0, 0, r.GetRows() - 1, t, 0, 0, t.GetRows() - 1);

            //gamma = t'rs / t't
            gamma = Compute_Inner_Product(rs, 0, 0, rs.GetRows() - 1, t, 0, 0, t.GetRows() - 1) / Compute_Inner_Product(t, 0, 0, t.GetRows() - 1, t, 0, 0, t.GetRows() - 1);

            DoubleComplex temp_gamma = make_cuDoubleComplex(1, 0) - gamma;

            //rs = rs - gamma*(rs - r)
            Compute_Vector_Linear_Combination(temp_gamma, rs, 0, 0, rs.GetRows() - 1, gamma, r, 0, 0, r.GetRows() - 1, rs, 0, 0, rs.GetRows() - 1);

            //xs = xs - gamma*(xs - x)
            Compute_Vector_Linear_Combination(temp_gamma, xs, 0, 0, xs.GetRows() - 1, gamma, x, 0, 0, x.GetRows() - 1, xs, 0, 0, xs.GetRows() - 1);

            //normr = ||rs||
            normr = Compute_L2Norm(rs, 0, 0, rs.GetRows() - 1);
        }
        else
        { //normr = ||r||
            normr = Compute_L2Norm(r, 0, 0, r.GetRows() - 1);
        }


        relres = normr / normb;
       // std::cout << "\nRel Residual Norm is:" << relres << " after iteration : " << this->num_iter + 1;

        cudaEventRecord(tempo2);
        cudaEventSynchronize(tempo2);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, tempo2);
       //  std::cout << "\nTiming:" << milliseconds;
        this->resvec.push_back({ normr , milliseconds }); //store timings and normr


        this->num_iter++;
        this->full_cycle++;

        if (relres <= this->rtol || normr < this->atol) //check convergence
        {
            this->info = SOLVER_INFO::SUCCESS;
            break;
        }


    }


    if (smoothing_operation == 1)
    {
        x = std::move(xs); //move xs to x
        r = std::move(rs); //move rs to r
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float  milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    this->runtime_milliseconds = milliseconds;
    //-------------------------------------------------stop time----------------------------------------------------------

    //get final stats
    this->iter_residual = normr;
    Compute_Residual(A, b, x, r);
    residual = Compute_L2Norm(r, 0, 0, r.GetRows() - 1);
    this->final_residual = residual;
    x.CopyMatrix_gpu_to_cpu();

    //set solver conclusion
    if (this->info != SOLVER_INFO::SUCCESS && this->info != SOLVER_INFO::DIVERGENCE)
    {
        if (this->init_residual > this->final_residual)
            this->info = SOLVER_INFO::SLOW_CONVERGENCE;
    }


}
