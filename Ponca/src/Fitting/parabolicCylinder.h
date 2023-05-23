#pragma once

#include "./defines.h"

#include <Eigen/Dense>


namespace Ponca
{

/*!
    \brief 
    \see 
*/
template < class DataPoint, class _WFunctor, typename T >
class ParabolicCylinder: public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    // PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        Check = Base::PROVIDES_PRIMITIVE_BASE && Base::PROVIDES_PLANE && Base::PROVIDES_TANGENT_PLANE_BASIS, /*!< \brief Requires PrimitiveBase and plane*/
//        PROVIDES_NORMAL_DERIVATIVE
                // PROVIDES_ALGEBRAIC_PARABOLIC_CYLINDER        /*!< \brief Provides Algebraic Parabolic Cylinder */
    };

public:
    using SampleMatrix  = Eigen::Matrix<Scalar, Eigen::Dynamic, 7>;
    using SampleVector  = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix2       = Eigen::Matrix<Scalar, 2, 2>;
    using Vector2       = Eigen::Matrix<Scalar, 2, 1>;
    using Vector7       = Eigen::Matrix<Scalar, 7, 1>;
// results
protected:
    Scalar        m_uc     {Scalar(0)};         /*!< \brief Constant parameter of the Algebraic hyper-parabolic cylinder */
    Vector2       m_ul     {Vector2::Zero()};   /*!< \brief Linear parameter of the Algebraic hyper-parabolic cylinder  */
    Scalar        m_a      {Scalar(0)};         /*!< \brief parameter of the Quadratic Parameter the Algebraic hyper-parabolic cylinder */
    Vector2       m_uq     {Vector2::Zero()};   /*!< \brief Quadratic parameter of the Algebraic hyper-parabolic cylinder  */

    Matrix2       m_H      {Matrix2::Zero()};

    SampleMatrix m_A_cov;
    SampleVector m_F_cov;


    bool m_planeIsReady {false};
public:
    
    PONCA_EXPLICIT_CAST_OPERATORS(ParabolicCylinder,parabolicCylinder)
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

    PONCA_MULTIARCH inline Scalar eval_quadratic_function(Scalar q1, Scalar q2) const {    
        Vector2 q {q1, q2};
        Scalar first = m_uc + m_ul.transpose() * q;
//        Scalar second = q.transpose() * m_H * q;
        Scalar product = m_uq.transpose() * q;
        Scalar second = m_a * product * product;
        return first + second ;
    } 

    //! \brief Value of the scalar field at the location \f$ \mathbf{q} \f$
    PONCA_MULTIARCH inline Scalar potential (const VectorType& _q) const;

    //! \brief Project a point on the ellipsoid
    PONCA_MULTIARCH inline VectorType project (const VectorType& _q) const;

    //! \brief Approximation of the scalar field gradient at \f$ \mathbf{q} (not normalized) \f$
     PONCA_MULTIARCH inline VectorType primitiveGradient (const VectorType& _q) const;

    /*! \brief Approximation of the scalar field gradient at the evaluation point */
    PONCA_MULTIARCH inline VectorType primitiveGradient () const {
        VectorType out {0, m_ul(0), m_ul(1)};
        return Base::tangentPlaneToWorld(out);
    }

//    MatrixType dNormal() const {
//        MatrixType matuq {0, 0        , 0
//                          0, m_uq(0,1), m_uq(0,1),
//                          0, m_uq(1,0), m_uq(1,1)};
//        return 2 * m_a * (matuq * matuq.transpose());}

//    // f(x) = uc + ul^T x + x^T Uq x
//    //      = uc + ul^T x + x^T P D P^T x
//    //      = uc + ul^T P (P^T x) + (P^T x)^T D (P^T x)
//    //      = uc + ul'^T y + y^T D y
//     std::pair<Vector2,Vector2> canonical() const
//     {
//         Eigen::SelfAdjointEigenSolver<Matrix2> eig(m_uq);
//         return std::make_pair(
//             m_ul.transpose() * eig.eigenvectors(),
//             eig.eigenvalues());
//     }

    PONCA_MULTIARCH inline VectorType plane_project (const VectorType& _q) const;

private:

    PONCA_MULTIARCH inline int    all_minimisation    ();
    PONCA_MULTIARCH inline int    first_minimisation  ();
    PONCA_MULTIARCH inline int    second_minimisation ();
    PONCA_MULTIARCH inline void    third_minimisation  ();
    
}; //class ParabolicCylinder

/// \brief Helper alias for ParabolicCylinder fitting on points
//! [ParabolicCylinderFit Definition]
template < class DataPoint, class _WFunctor, typename T>
    using ParabolicCylinderFit =
    Ponca::ParabolicCylinder<DataPoint, _WFunctor,
        Ponca::CovariancePlaneFitImpl<DataPoint, _WFunctor,
                Ponca::CovarianceFitBase<DataPoint, _WFunctor,
                        Ponca::MeanPosition<DataPoint, _WFunctor,
                                Ponca::Plane<DataPoint, _WFunctor,
                                    Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>;
//! [ParabolicCylinderFit Definition]

#include "parabolicCylinder.hpp"
}
