/*!
\file Preconditioner.h
 \brief Definition of classes Preconditioner, JacobiPreconditioner, RichardsonPreconditioner and prototypes of other functions related to preconditioners 
*/
# pragma once

#include <cuComplex.h>


#include"location_enums.h"

class CSR_Matrix;
class Dense_Matrix;
typedef cuDoubleComplex DoubleComplex;


//! enum class which the type of the preconditioner
enum class PRECONDITIONER_TYPE {
    JACOBI = 0, /*!< Jacobi preconditioner */
    RICHARDSON =1 /*!< Richardson preconditioner*/
};

//! abstract class which serves as base for any specific preconditioner object 
class Preconditioner {
private:
    const PRECONDITIONER_TYPE precond_type; /*!< stores the type of the preconditioner*/
protected:
    CPU_EXISTENCE cpu_exists = CPU_EXISTENCE::NON_EXISTENT; /*!< indicates whether or not the preconditioner object's internal arrays exist on CPU memory*/
    GPU_EXISTENCE gpu_exists = GPU_EXISTENCE::NON_EXISTENT;/*!< indicates whether or not the preconditioner object's internal arrays  exist on GPU memory*/
public:

    //! Parametrized constructor for the abstract class
    /*!
     Initializes the preconditioner type
     \param[in] type the preconditioner type
    */
    Preconditioner(PRECONDITIONER_TYPE type) : precond_type{ type }
    {
    }

    
    //! Returns true if preconditioner object arrays exist on CPU memory
    /*!
    \return boolen value
    */
    bool Exists_cpu() const
    {
        return cpu_exists == CPU_EXISTENCE::EXISTENT;
    }



     //! Returns true if preconditioner object arrays exist on GPU memory
    /*!
    \return boolean value
    */
    bool Exists_gpu() const
    {
        return gpu_exists == GPU_EXISTENCE::EXISTENT;
    }
   
   

   //! Returns the type of the preconditioner
    /*!
    \return preconditioner type
    */
    PRECONDITIONER_TYPE GetPrecondType() const {
        return precond_type;
    }


    //! Pure virtual member function which is used for applying preconditioner polymorphically  
    virtual void ApplyPreconditioner(const Dense_Matrix& vec, Dense_Matrix& result)  const = 0;

    //! Desturctor for Preconditioner class
    virtual ~Preconditioner() { };
};





//! JacobiPreconditioner class derived from base class Preconditioner
class JacobiPreconditioner : public Preconditioner {
private:
    const int diag_length; /*!< length of the array containing the Jacobi Preconditioner values */
    DoubleComplex* gpu_d_inverse = nullptr; /*!< pointer to array on GPU containing the preconditioner values; 
    it is equal to nullptr in case no such array exists. Note:values on CPU and GPU do not match until they are copied.*/
    DoubleComplex* cpu_d_inverse = nullptr; /*!< pointer to array on CPU containing the preconditioner values;
    it is equal to nullptr in case no such array exists. Note:values on CPU and GPU do not match until they are copied.. */


    int Initialize_Preconditioner(const CSR_Matrix& A);

public:
    JacobiPreconditioner(const CSR_Matrix& A);

    void Allocate_Memory(const LOCATION loc);

    void Deallocate_Memory(const LOCATION loc);

    ~JacobiPreconditioner();

   
   //! returns pointer to array on GPU containing the preconditioner values
    /*!
    \return  pointer to array on GPU containing the preconditioner values;nullptr in case no memory is allocated
    */
    DoubleComplex* Get_GPU_d_inverse() const {
        if(Exists_gpu() == true)
          return gpu_d_inverse;
        else
          return nullptr;
    }


    //! returns pointer to array on CPU containing the preconditioner values
    /*!
    \return  pointer to array on CPU containing the preconditioner values;nullptr in case no memory is allocated
    */
    DoubleComplex* Get_CPU_d_inverse() const {
        if(Exists_cpu() == true)
           return cpu_d_inverse;
        else
          return nullptr;
    }


    //! returns length of the array storing preconditioner values
    int Get_Diag_Length() const
    {
        return diag_length;
    }

    void CopyPreconditioner_cpu_to_gpu() ;

    void CopyPreconditioner_gpu_to_cpu() ;

    void ApplyPreconditioner(const Dense_Matrix& vec, Dense_Matrix& result) const override;

    //! Move constructor for Jacobi Preconditioner class
    /*!
     A deleted constructor
     */
    JacobiPreconditioner(JacobiPreconditioner&&) = delete;

     //! Copy constructor for Jacobi Preconditioner class
    /*!
     A deleted constructor
     */
    JacobiPreconditioner(const JacobiPreconditioner&) = delete;

    //! Copy assignment operator for Jacobi Preconditioner class
    /*!
     A deleted operator
     */
    JacobiPreconditioner& operator= (JacobiPreconditioner&&) = delete;


     //! Move assignment operator for Jacobi Preconditioner class
    /*!
     A deleted operator
     */
    JacobiPreconditioner& operator= (const JacobiPreconditioner&) = delete;

};











//! RichardsonPreconditioner class derived from base class Preconditioner
class RichardsonPreconditioner : public Preconditioner{
  
 private:
    const int Identity_dim; /*!< length of the array containing the Richardson Preconditioner values */

 public:

   //! returns dimension of richardson/identity preconditioner
   int Get_Identity_Dimension() const
   {
     return Identity_dim;
   }

   RichardsonPreconditioner(const CSR_Matrix& A);
   void ApplyPreconditioner(const Dense_Matrix& vec, Dense_Matrix& result)  const override;

   //! Default destructor for Richardson preconditioner 
   ~RichardsonPreconditioner()
   {

   }
 
};

Preconditioner* Generate_Preconditioner(const PRECONDITIONER_TYPE precond_type, const CSR_Matrix& A);









