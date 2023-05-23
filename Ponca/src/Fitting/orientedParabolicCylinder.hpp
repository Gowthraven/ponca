#include <Eigen/Geometry>

namespace Internal
{

//!
//! \brief solve DX + XD = C
//! \param D a vector of size N representing a NxN diagonal matrix
//! \param C any NxN matrix
//!
template<typename MatrixType, class VectorType>
MatrixType solve_diagonal_sylvester(const VectorType& D, const MatrixType& C)
{
    using Index = typename VectorType::Index;
    using Scalar = typename VectorType::Scalar;

    const Index n = D.size();
    const Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    MatrixType X;
    for(Index i = 0; i < n; ++i)
    {
        for(Index j = 0; j < n; ++j)
        {
            const Scalar d = D(i) + D(j);
            if(abs(d) < epsilon)
            {
                X(i,j) = Scalar(0);
            }
            else
            {
                X(i,j) = C(i,j) / d;
            }
        }
    }
    return X;
}

//!
//! \brief solve AX + XA = C
//! \param A a symmetric NxN matrix
//! \param C any NxN matrix
//!
template<typename MatrixType>
MatrixType solve_symmetric_sylvester(const MatrixType& A, const MatrixType& C)
{
    Eigen::SelfAdjointEigenSolver<MatrixType> eig(A);
    const auto& D = eig.eigenvalues();
    const MatrixType& P = eig.eigenvectors();
    const MatrixType Pinv = P.transpose();
    const MatrixType F = Pinv * C * P;
    const MatrixType Y = solve_diagonal_sylvester(D, F);
    const MatrixType X = P * Y * Pinv;
    return X;
}

template<typename MatrixType>
MatrixType solve_symmetric_sylvester_2d(const MatrixType& A, const MatrixType& C)
{
    using Scalar = typename MatrixType::Scalar;
    using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
    using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
    if constexpr (A.rows() == 2)
    {
        Matrix4 M = Matrix4::Zero();
        M(0,0) = 2 * A(0,0); // 2*a
        M(0,1) = 2 * A(1,0); // 2*b
        M(1,0) = A(1,0); // b
        M(1,1) = A(0,0) + A(1,1); // a + d
        M(1,2) = A(1,0); // b
        M(2,1) = 2 * A(1,0); // 2*b
        M(2,2) = 2 * A(1,1); // 2*d

        Vector4 b;
        b[0] = C(0,0);
        b[1] = C(1,0);
        b[2] = C(1,1);

        // solve Mx = b
        const Vector4 x = M.colPivHouseholderQr().solve(b);

        MatrixType sol;
        sol(0,0) = x[0];
        sol(1,0) = x[1];
        sol(0,1) = x[1];
        sol(1,1) = x[2];

        return sol;
    }
    else
    {
        return MatrixType::Zero();
    }
}
} // namespace Internal

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    m_ul.setZero();
    m_uq.setZero();
    m_a = Scalar(0);
    m_uc = Scalar(0);

    // 2D data
    m_sumN2D.setZero();
    m_sumP2D.setZero();
    m_sumDotPN2D = Scalar(0);
    m_sumDotPP2D = Scalar(0);
    m_prodPP2D.setZero();
    m_prodPN2D.setZero();

    m_planeIsReady = false;
}

template < class DataPoint, class _WFunctor, typename T>
bool
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                      const VectorType &localQ,
                                                      const DataPoint &attributes)
{
    if(! m_planeIsReady)
    {
        return Base::addLocalNeighbor(w, localQ, attributes);
    }
    else // base plane is ready, we can now fit the patch
    {
        // express neighbor in local coordinate frame
        VectorType localPos = Base::worldToTangentPlane(attributes.pos());
        Vector2 planePos = Vector2 ( *(localPos.data()+1), *(localPos.data()+2) );

        VectorType localNorm =  Base::template worldToTangentPlane<true>(attributes.normal());
        Vector2 planeNorm = Vector2 ( *(localNorm.data()+1), *(localNorm.data()+2) );

        m_sumN2D     += w * planeNorm;
        m_sumP2D     += w * planePos;
        m_sumDotPN2D += w * planeNorm.dot(planePos);
        m_sumDotPP2D += w * planePos.squaredNorm();
        m_prodPP2D   += w * planePos * planePos.transpose();
        m_prodPN2D   += w * planePos * planeNorm.transpose();

        return true;
    }
}

template < class DataPoint, class _WFunctor, typename T>
typename OrientedParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::potential( const VectorType &_q ) const
{  
    VectorType x = Base::worldToTangentPlane(_q);
    return eval_quadratic_function(*(x.data() +1 ), *(x.data() + 2));
}


template < class DataPoint, class _WFunctor, typename T>
typename OrientedParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::project( const VectorType& _q ) const
{   
    VectorType x = Base::worldToTangentPlane(_q);
    return Base::project(_q) + (eval_quadratic_function(*(x.data()+1), *(x.data()+2)) * Base::primitiveGradient(_q));
}

template < class DataPoint, class _WFunctor, typename T>
typename OrientedParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::plane_project( const VectorType& _q ) const
{   
    return Base::project(_q);
}

template < class DataPoint, class _WFunctor, typename T>
typename OrientedParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::primitiveGradient( const VectorType& _q ) const
{
    // Convexe = m_a >= 0    Concave = m_a <= 0

    VectorType proj = Base::worldToTangentPlane(_q);
    Vector2 temp {proj(1),  proj(2)};
    Vector2 df = m_ul + 2 * m_a * (m_uq * m_uq.transpose()) * temp;
//    Vector2 df = m_ul + 2 * m_H * temp; // => Only used if you want to try only first and second minimisation.

    VectorType local_gradient { 1, -df(0) , -df(1) };
    local_gradient.normalize();

    VectorType world_gradient = Base::m_solver.eigenvectors() * local_gradient;
    world_gradient.normalize();

    return world_gradient;
}


template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::finalize () {

    if (! m_planeIsReady) {
        FIT_RESULT res = Base::finalize();

        if (res == STABLE) {
            m_planeIsReady = true;
            return Base::m_eCurrentState = NEED_OTHER_PASS;
        }
        return res;
    }
    else {
        int res = all_minimisation();
        if (res == 0) return Base::m_eCurrentState = UNSTABLE;
        return Base::m_eCurrentState = STABLE;
    }
}

template < class DataPoint, class _WFunctor, typename T>
int
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::all_minimisation () {
    first_minimisation();
    second_minimisation();
    third_minimisation();
    return 1;
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::first_minimisation () {
    PONCA_MULTIARCH_STD_MATH(abs);
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();
    const Scalar invSumW = Scalar(1.)/Base::m_sumW;

    const Matrix2 A = 2 * (Base::m_sumW * m_prodPP2D -  m_sumP2D * m_sumP2D.transpose());
    Matrix2 C = Base::m_sumW * m_prodPN2D - m_sumP2D * m_sumN2D.transpose();
    C = C + C.transpose().eval();

    m_H = internal::solve_symmetric_sylvester(A, C) / 2;
    m_ul = invSumW * (m_sumN2D - Scalar(2) * m_H * m_sumP2D);
    m_uc = - invSumW * ( m_ul.dot(m_sumP2D) + (m_prodPP2D * m_H).trace() );

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(m_H);
    Vector2 values = eig.eigenvalues();

    int higher = abs(values(0)) > abs(values(1)) ? 0 : 1;

    Scalar lambda0 = values(higher);
    Scalar lambda1 = values((higher + 1) % 2);
    Scalar t = Base::m_w.evalScale();

    Scalar alpha = 1;
    if (abs(lambda0 + 1/t) > epsilon)
        alpha = 2 * (abs(lambda0) - abs(lambda1)) / (abs(lambda0) + 1 / t);
    m_a = ( alpha < 1 ) ? alpha : Scalar(1);
    const Eigen::MatrixXd eigenVec = eig.eigenvectors();

    m_uq = eigenVec.col(higher);

}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::second_minimisation () { 
    const Scalar invSumW = Scalar(1.)/Base::m_sumW;

    Matrix2 Q = 2 * m_uq * m_uq.transpose();                   // 2x2
    Matrix2 Q_squared = Q * Q;                                 // 2x2
    Scalar A = (m_prodPP2D.array() * Q_squared.array()).sum(); // 1
    Vector2 B = Q * m_sumP2D;                                  // 2x2 * 2x1 = 2x1
    Scalar C = (m_prodPN2D.array() * Q.array()).sum();         // 1

    Scalar first = invSumW * m_sumN2D.transpose() * B;         // 1 * 1x2 * 2x1 = 1
    Scalar second = invSumW * B.transpose() * B;               // 1 * 1x2 * 2x1 = 1

    Scalar a = (C - first) / (4 * A - second);                 // (1 - 1) / (1 * 1 - 1)
    m_a *= a;

    std::cout << "Computed a with der on a : " << m_a << " " << std::endl;



}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinder<DataPoint, _WFunctor, T>::third_minimisation () {
    const Scalar invSumW = Scalar(1.)/Base::m_sumW;
    Matrix2 Q = 2 * m_uq * m_uq.transpose();
    Vector2 B = m_a * Q * m_sumP2D;

    Scalar C = m_a * (m_prodPN2D.array() * Q.array()).sum();
    Scalar A = m_ul.transpose() * m_sumP2D;

    m_ul = invSumW * ( m_sumN2D - B);

    m_uc = - invSumW * (A + C);
}
