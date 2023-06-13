
namespace internal {

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

//!
//! \brief solve AX + XA = C
//! \param A a symmetric NxN matrix
//! \param C a symmetric NxN matrix
//!
template<typename MatrixType>
MatrixType solve_symmetric_sylvester_2(const MatrixType& A, const MatrixType& C)
{
    using Scalar = typename MatrixType::Scalar;
    using Matrix6 = Eigen::Matrix<Scalar, 6, 6>;
    using Vector6 = Eigen::Matrix<Scalar, 6, 1>;
    if constexpr (A.rows() == 3)
    {
        Matrix6 M = Matrix6::Zero();
        M(0,0) = 2 * A(0,0); // 2 a
        M(0,1) = 2 * A(1,0); // 2 b
        M(0,2) = 2 * A(2,0); // 2 c
        M(1,0) = A(1,0); // b
        M(1,1) = A(0,0) + A(1,1); // a + d
        M(1,2) = A(2,1); // e
        M(1,3) = A(1,0); // b
        M(1,4) = A(2,0); // c
        M(2,0) = A(2,0); // c
        M(2,1) = A(2,1); // e
        M(2,2) = A(0,0) + A(2,2); // a + f
        M(2,4) = A(1,0); // b
        M(2,5) = A(2,0); // c
        M(3,1) = 2 * A(1,0); // 2 b
        M(3,3) = 2 * A(1,1); // 2 d
        M(3,4) = 2 * A(2,1); // 2 e
        M(4,1) = A(2,0); // c
        M(4,2) = A(1,0); // b
        M(4,3) = A(2,1); // e
        M(4,4) = A(1,1) + A(2,2); // d + f
        M(4,5) = A(2,1); // e
        M(5,2) = 2 * A(2,0); // 2 c
        M(5,4) = 2 * A(2,1); // 2 e
        M(5,5) = 2 * A(2,2); // 2 f

        Vector6 b;
        b[0] = C(0,0);
        b[1] = C(1,0);
        b[2] = C(2,0);
        b[3] = C(1,1);
        b[4] = C(2,1);
        b[5] = C(2,2);

        // solve Mx = b
        const Vector6 x = M.colPivHouseholderQr().solve(b);

        MatrixType sol;
        sol(0,0) = x[0];
        sol(1,0) = x[1];
        sol(0,1) = x[1];
        sol(2,0) = x[2];
        sol(0,2) = x[2];
        sol(1,1) = x[3];
        sol(2,1) = x[4];
        sol(1,2) = x[4];
        sol(2,2) = x[5];

        return sol;
    }
    else
    {
        return MatrixType::Zero();
    }
}

} // namespace internal

template < class DataPoint, class _WFunctor, typename T>
void
OrientedEllipsoidFitImpl<DataPoint, _WFunctor, T>::init(
    const VectorType& _evalPos)
{
    Base::init(_evalPos);

    // Setup fitting internal values
    m_sumDotPN = Scalar(0.0);
    m_sumDotPP = Scalar(0.0);
    m_sumProdPP = MatrixType::Zero();
    m_sumProdPN = MatrixType::Zero();
    m_A = MatrixType::Zero();
}

template < class DataPoint, class _WFunctor, typename T>
bool
OrientedEllipsoidFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(
    Scalar w,
    const VectorType &localQ,
    const DataPoint &attributes)
{
    if( Base::addLocalNeighbor(w, localQ, attributes) )
    {
        m_sumDotPN += w * attributes.normal().dot(localQ);
        m_sumDotPP += w * localQ.squaredNorm();
        m_sumProdPP += w * localQ * localQ.transpose();
        m_sumProdPN += w * localQ * attributes.normal().transpose();
        return true;
    }
    return false;
}


template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
OrientedEllipsoidFitImpl<DataPoint, _WFunctor, T>::finalize()
{
    // handle specific configurations
    // With less than 3 neighbors the fitting is undefined
    if(Base::finalize() != STABLE || Base::m_nbNeighbors < 3)
    {
        return Base::m_eCurrentState;
    }

    m_A = 2 * (Base::m_sumW * m_sumProdPP -  Base::m_sumP * Base::m_sumP.transpose());
    MatrixType C = Base::m_sumW * m_sumProdPN - Base::m_sumP * Base::m_sumN.transpose();
    C = C + C.transpose().eval();

    const Scalar invSumW = Scalar(1.)/Base::m_sumW;

    Base::m_uq = internal::solve_symmetric_sylvester(m_A, C);
    Base::m_ul = invSumW * (Base::m_sumN - Scalar(2) * Base::m_uq * Base::m_sumP);
    Base::m_uc = - invSumW * ( Base::m_ul.dot(Base::m_sumP) + (m_sumProdPP * Base::m_uq).trace() );

    Base::m_eCurrentState = STABLE;
    return Base::m_eCurrentState;
}

template < class DataPoint, class _WFunctor, int DiffType, typename T>
void
OrientedEllipsoidDerImpl<DataPoint, _WFunctor, DiffType, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    m_dSumDotPN.setZero();
    m_dSumDotPP.setZero();
    m_dSumProdPN.setZero();
    m_dSumProdPP.setZero();

    m_dUc.setZero();
    m_dUq.setZero();
    m_dUl.setZero();
}


template < class DataPoint, class _WFunctor, int DiffType, typename T>
bool
OrientedEllipsoidDerImpl<DataPoint, _WFunctor, DiffType, T>::addLocalNeighbor(Scalar w,
                                                                           const VectorType &localQ,
                                                                           const DataPoint &attributes,
                                                                           ScalarArray &dw) {
    if( Base::addLocalNeighbor(w, localQ, attributes, dw) ) {

        m_dSumDotPN += dw * attributes.normal().dot(localQ);
        m_dSumDotPP += dw * localQ.squaredNorm();
        for(auto i = 0; i < Dim; ++i) {
            m_dSumProdPP.template block<Dim,Dim>(0, i*Dim) += dw[i] * localQ * localQ.transpose();
            m_dSumProdPN.template block<Dim,Dim>(0, i*Dim) += dw[i] * localQ * attributes.normal().transpose();
        }

        return true;
    }
    return false;
}


template < class DataPoint, class _WFunctor, int DiffType, typename T>
FIT_RESULT
OrientedEllipsoidDerImpl<DataPoint, _WFunctor, DiffType, T>::finalize()
{
    PONCA_MULTIARCH_STD_MATH(sqrt);

    Base::finalize();
    // Test if base finalize end on a viable case (stable / unstable)
    if (this->isReady())
    {
        for(auto i = 0; i < Dim; ++i) {
            const MatrixType dA = 2 * (
                  Base::m_dSumW[i] * Base:: m_sumProdPP
                + Base::m_sumW * m_dSumProdPP.template block<Dim,Dim>(0, i*Dim)
                - Base::m_dSumP.col(i) * Base::m_sumP.transpose()
                - Base::m_sumP * Base::m_dSumP.col(i).transpose());
            MatrixType dC =
                  Base::m_dSumW[i] * Base::m_sumProdPN
                + Base::m_sumW * m_dSumProdPN.template block<Dim,Dim>(0, i*Dim)
                - Base::m_dSumP.col(i) * Base::m_sumN.transpose()
                - Base::m_sumP * Base::m_dSumN.col(i).transpose();
            dC = dC + dC.transpose().eval();

            const MatrixType C = dC - dA * Base::m_uq - Base::m_uq * dA;
            m_dUq.template block<Dim,Dim>(0, i*Dim) = internal::solve_symmetric_sylvester(Base::m_A, C);
        }

        const Scalar invSumW = Scalar(1) / Base::m_sumW;

        for(auto i = 0; i < Dim; ++i) {
            m_dUl.col(i) = invSumW * (
                Base::m_dSumN.col(i) - 2 * (
                    m_dUq.template block<Dim,Dim>(0, i*Dim) * Base::m_sumP
                    + Base::m_uq * Base::m_dSumP.col(i) )
                - Base::m_dSumW[i] * Base::m_ul);
        }

        m_dUc = ScalarArray::Zero(); 

        for(auto i = 0; i < Dim; ++i) {
            MatrixType traceTerm = m_dSumProdPP.template block<Dim,Dim>(0, i*Dim) * Base::m_uq 
                                + Base::m_sumProdPP * m_dUq.template block<Dim,Dim>(0, i*Dim);

            Scalar traceValue = traceTerm.trace();    

            m_dUc[i] = -invSumW * (
                  m_dUl.col(i).dot(Base::m_sumP)
                + Base::m_ul.dot(Base::m_dSumP.col(i))
                + traceValue
                - Base::m_dSumW[i] * Base::m_uc);
}

    }

    return Base::m_eCurrentState;
}

template < class DataPoint, class _WFunctor, int DiffType, typename T>
typename OrientedEllipsoidDerImpl<DataPoint, _WFunctor, DiffType, T>::MatrixArray
OrientedEllipsoidDerImpl<DataPoint, _WFunctor, DiffType, T>::dShapeOperator3D() const
{
    MatrixArray dS;
    for(auto i = 0; i < Dim; ++i)
    {
        dS.template block<Dim,Dim>(0, i*Dim) =
            2 * (
                Base::m_ul.norm() * m_dUq.template block<Dim,Dim>(0, i*Dim)
                - Base::m_uq * Base::m_ul.normalized().dot(m_dUl.col(i))
            ) / Base::m_ul.squaredNorm();
    }
    return dS;
}
