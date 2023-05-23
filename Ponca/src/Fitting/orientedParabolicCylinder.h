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
class OrientedParabolicCylinder: public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    // PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        Check = Base::PROVIDES_PRIMITIVE_BASE &&  Base::PROVIDES_PLANE //, /*!< \brief Requires PrimitiveBase and plane*/
        // PROVIDES_ALGEBRAIC_PARABOLIC_CYLINDER        /*!< \brief Provides Algebraic Parabolic Cylinder */
    };

public:
    using Matrix2       = Eigen::Matrix<Scalar, 2, 2>;
    using Vector2       = Eigen::Matrix<Scalar, 2, 1>;
// results
protected:

    Scalar        m_uc     {Scalar(0)};         /*!< \brief Constant parameter of the Algebraic hyper-parabolic cylinder */
    Vector2       m_ul     {Vector2::Zero()};   /*!< \brief Linear parameter of the Algebraic hyper-parabolic cylinder  */
    Scalar        m_a      {Scalar(0)};         /*!< \brief parameter of the Quadratic Parameter the Algebraic hyper-parabolic cylinder */
    Vector2       m_uq     {Vector2::Zero()};   /*!< \brief Quadratic parameter of the Algebraic hyper-parabolic cylinder  */
    Matrix2       m_H      {Matrix2::Zero()};

        // computation data

    // 2D data
    Vector2    m_sumN2D,      /*!< \brief Sum of the normal vectors */
               m_sumP2D;      /*!< \brief Sum of the relative positions */
    Scalar     m_sumDotPN2D,  /*!< \brief Sum of the dot product betwen relative positions and normals */
               m_sumDotPP2D;  /*!< \brief Sum of the squared relative positions */
    Matrix2    m_prodPP2D,    /*!< \brief Sum of exterior product of positions */
               m_prodPN2D;    /*!< \brief Sum of exterior product of positions and normals */

    bool m_planeIsReady {false};

public:
    
    PONCA_EXPLICIT_CAST_OPERATORS(OrientedParabolicCylinder,orientedParabolicCylinder)
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

    PONCA_MULTIARCH inline Scalar eval_quadratic_function(Scalar q1, Scalar q2) const {    
        Vector2 q {q1, q2};
        Scalar first = m_uc + m_ul.transpose() * q;
//         Scalar second = q.transpose() * m_H * q;
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

    PONCA_MULTIARCH inline VectorType plane_project (const VectorType& _q) const;
private:

    PONCA_MULTIARCH inline int    all_minimisation    ();
    PONCA_MULTIARCH inline void    first_minimisation  ();
    PONCA_MULTIARCH inline void    second_minimisation ();
    PONCA_MULTIARCH inline void    third_minimisation  ();


}; //class ParabolicCylinder

/// \brief Helper alias for ParabolicCylinder fitting on points
//! [ParabolicCylinderFit Definition]
template < class DataPoint, class _WFunctor, typename T>
    using OrientedParabolicCylinderFit =
    Ponca::OrientedParabolicCylinder<DataPoint, _WFunctor,
        Ponca::CovariancePlaneFitImpl<DataPoint, _WFunctor,
                        Ponca::CovarianceFitBase<DataPoint, _WFunctor,
                                Ponca::MeanPosition<DataPoint, _WFunctor,
                                        Ponca::Plane<DataPoint, _WFunctor,
                                            Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>;
//! [ParabolicCylinderFit Definition]

#include "orientedParabolicCylinder.hpp"
}
