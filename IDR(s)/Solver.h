/*!
\file Solver.h
\brief Definition of class Solver and enum and structs related to it 
*/
#pragma once

#include<vector>
#include<string>

#include<cuComplex.h>

typedef cuDoubleComplex DoubleComplex;
class CSR_Matrix;
class Dense_Matrix;
class Preconditioner;

//! An enum class which describes the different states the solver may end up in
enum class SOLVER_INFO {
    UNALLOCATED_INPUT_MATRICES, /*!< indicates error when input matrices are not allocated*/

    NOT_CONVERGED, /*!< typically used as the initial state of the solver and indicates solution has not yet converged */

    SUCCESS, /*!< indicates solution has converged successfully, which means that an iteratively computed residual
     satisfying the tolerance conditions is obtained within the limit of  maximum number of iterations.
      However, it does not make  any comment about the accuracy of the final solution that is whether
       the true residual satisfies the tolerance conditions or not. */

    ERR_NOT_SUPPORTED, /*!<  indicates error when the input matrix is not supported by a particular solver*/

    DIVERGENCE, /*!< indicates solution has diverged */

    SLOW_CONVERGENCE /*!< indicates slow convergence: meaning iteration count has exceeded max iterations and 
    convergence is not yet achieved. But, true residual at end of the iterations is less than the starting residual, 
    so solution is hoped to converge if more iterations are allowed.*/
};



//! An enum class which describes the name of the different solvers available to solve the linear system
enum class SOLVER_NAME {
    PIDR, /*!< Preconditioned Induced Dimension Reduction krylov solver */
};


//! A structure which is used to store information about the iteratively computed resdiuals and the timings when they are computed
struct resInfo {
    double normr; /*!<  the iteratively computed residual*/
    double timing; /*!< time when the residual is computed(this is actually the time difference as the time for the initial residual is taken as 0.0)*/
};

typedef struct resInfo resInfo;



//! This class contains various solver parameters and  solver(s) as member fuction(s)
/*! 
 The solver parameters(like relative, absolute tolerance, max_iter etc.) can be set by the user.
  A member function is invoked to solve the linear system.
*/
class Solver {

private:
    SOLVER_NAME name; /*!< name of the solver invoked */
    SOLVER_INFO info = SOLVER_INFO::NOT_CONVERGED; /*!< maintains information about the solver state*/
    int num_iter = 0; /*!< the number of iterations occurred to solve the linear system */
    int spmv_count = 0; /*!< number of sparse matrix vector products occurred while solving the linear system */
    double init_residual = 0.0; /*!< the initial residual */
    double iter_residual = 0.0; /*!< the iteratively computed residual at the end */
    double final_residual = 0.0; /*!< true residual at the end */
    double runtime_milliseconds = 0.0; /*!< time(in milliseconds) to get the final solution*/
    std::vector<resInfo> resvec; /*!< vector containing information about the iteratively computed residuals and the timings when they are computed */
    int full_cycle = 0; /*! total number of full cycles ocurred to solve the linear system (useful in algorithms like IDR)*/
    
    int max_iter = 0; /*!<  maximum number of iterations allowed; set by the algorithm or the user*/
    double atol = 0; /*!< absolute tolerance; set by the algorithm or the user */
    double rtol = 0; /*!< relative tolerance; set by the algorithm or the user*/

public:
 /*! returns name of the solver invoked */
    std::string GetSolverName()const
    {
          switch(name)
          {
            case SOLVER_NAME::PIDR:
             return "PIDR";
          }

          return NULL;
    }

    /*! returns information about the solver state*/
    std::string GetInfo() const 
    {
             switch(info)
        {
            case SOLVER_INFO::UNALLOCATED_INPUT_MATRICES:
              return "UNALLOCATED_INPUT_MATRICES";
              break;
            
            case SOLVER_INFO::NOT_CONVERGED:
              return "NOT_CONVERGED";
              break;

            case SOLVER_INFO::SUCCESS:
              return "SUCCESS";
              break;

            case SOLVER_INFO::ERR_NOT_SUPPORTED:
               return "ERR_NOT_SUPPORTED";
             break;

            case SOLVER_INFO::DIVERGENCE:
             return "DIVERGENCE";
             break;

            case SOLVER_INFO::SLOW_CONVERGENCE:
             return "SLOW_CONVERGENCE";
             break;

        }

        return NULL;
    }


    /*! returns the number of iterations occurred to solve the linear system */
    int GetNum_iter() const 
    {
       return num_iter;
    }

 /*! returns number of sparse matrix vector products occurred while solving the linear system */
    int GetSpmv_count() const
    {
        return spmv_count;
    }

     /*! the initial residual */
    double GetInit_residual()const
    {
        return init_residual;
    }

 /*! returns the iteratively computed residual at the end */
    double GetIter_residual() const
    {
        return iter_residual;
    }

/*! returns true residual at the end */
    double GetFinal_residual() const
    {
        return final_residual;
    }

/*! returns time(in milliseconds) to get the final solution*/
    double GetRuntimeMilliseconds()const
    {
        return runtime_milliseconds;
    }

/*! retruns the number of full cycles ocurred to solve the linear system (useful in algorithms like IDR)*/
    int GetFull_cycle()const
    {
        return full_cycle;
    }

 /*! returns vector containing information about the iteratively computed residuals and the timings when they are computed */
    std::vector<resInfo> GetResvec() const
    {
        return resvec;
    }

/*! returns absolute tolerance */
    double GetAtol() const
    {
        return atol;
    }

/*! set absolute tolerance */
    void SetAtol(const double atol)
    {
        this->atol = atol;
    }

/*! retruns relative tolerance*/
    double GetRtol() const
    {
        return rtol;
    }

/*! set relative tolerance*/
    void SetRtol(const double rtol)
    {
        this->rtol = rtol;
    }

/*!returns  maximum number of iterations allowed*/
    double GetMax_iter()const
    {
        return max_iter;
    }

/*! set max number of iterations to solve linear system */
    void SetMax_iter(const int max_iter)
    {
        this->max_iter = max_iter;
    }

    void PIDR_Solver(const CSR_Matrix& A, const Dense_Matrix& b, Dense_Matrix& x,const  Preconditioner& precond, const int s);


};

