#pragma once

#include "algebraicEllipsoid.h"
#include <Ponca/Fitting>

namespace Ponca
{

/*!
    \brief Algebraic Ellipsoid fitting procedure on oriented point sets

    \inherit Concept::FittingProcedureConcept

    \see AlgebraicEllipsoid
*/
template < class DataPoint, class _WFunctor, typename T >
class OrientedEllipsoidFitImpl : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE

 protected:
    enum
    {
        Check = Base::PROVIDES_ALGEBRAIC_ELLIPSOID and
                Base::PROVIDES_MEAN_NORMAL and
                Base::PROVIDES_MEAN_POSITION
    };

public:
    // computation data
    Scalar  m_sumDotPN,  /*!< \brief Sum of the dot product betwen relative positions and normals */
            m_sumDotPP;  /*!< \brief Sum of the squared relative positions */
    MatrixType m_sumProdPP, /*!< \brief Sum of exterior product of positions */
               m_sumProdPN; /*!< \brief Sum of exterior product of positions and normals */
    MatrixType m_A;

public:
    PONCA_EXPLICIT_CAST_OPERATORS(OrientedEllipsoidFitImpl,orientedEllipsoidFit)
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

}; //class OrientedEllipsoidFitImpl

template < class DataPoint, class _WFunctor, typename T>
using OrientedEllipsoidFit =
OrientedEllipsoidFitImpl<DataPoint, _WFunctor,
        MeanPosition<DataPoint, _WFunctor,
            MeanNormal<DataPoint, _WFunctor,
                AlgebraicEllipsoid<DataPoint, _WFunctor,T>>>>;

template < class DataPoint, class _WFunctor, int DiffType, typename T>
class OrientedEllipsoidDerImpl : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_DEFAULT_DER_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE
    enum {
        Dim = MatrixType::RowsAtCompileTime
    };
    using MatrixArray = Eigen::Matrix<Scalar,3,9>;

    // only work in 3D with spatial derivative
    static_assert(DiffType == Ponca::FitSpaceDer);
    static_assert(Dim == 3);

protected:
    enum
    {
        Check = Base::PROVIDES_ALGEBRAIC_ELLIPSOID and
                Base::PROVIDES_MEAN_POSITION_DERIVATIVE and
                Base::PROVIDES_MEAN_NORMAL_DERIVATIVE and
                Base::PROVIDES_PRIMITIVE_DERIVATIVE,
        PROVIDES_ELLIPSOID_SPHERE_DERIVATIVE
    };

    ScalarArray m_dSumDotPN,
                m_dSumDotPP;
    MatrixArray m_dSumProdPN,
                m_dSumProdPP;

public:
    ScalarArray m_dUc;
    VectorArray m_dUl;
    MatrixArray m_dUq;

public:
    PONCA_EXPLICIT_CAST_OPERATORS_DER(OrientedEllipsoidDerImpl,orientedEllipsoidDer)
    PONCA_FITTING_DECLARE_INIT_ADDDER_FINALIZE

    // shape op 3D = d2 f /dx = 2 uq / norm(ul)
    inline MatrixArray dShapeOperator3D() const;
};

template < class DataPoint, class _WFunctor, int DiffType, typename T>
using OrientedEllipsoidDer =
    OrientedEllipsoidDerImpl<DataPoint, _WFunctor, DiffType,
        MeanPositionDer<DataPoint, _WFunctor, DiffType,
            MeanNormalDer<DataPoint, _WFunctor, DiffType, T>>>;

#include "orientedEllipsoidFit.hpp"

} //namespace Ponca
