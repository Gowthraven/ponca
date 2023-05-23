template < class DataPoint, class _WFunctor, typename T>
typename AlgebraicEllipsoid<DataPoint, _WFunctor, T>::Scalar
AlgebraicEllipsoid<DataPoint, _WFunctor, T>::potential( const VectorType &_q ) const
{
    // turn to centered basis
    const VectorType lq = Base::m_w.convertToLocalBasis(_q);

    return m_uc + lq.dot(m_ul) + lq.transpose() * m_uq * lq;
}

template < class DataPoint, class _WFunctor, typename T>
typename AlgebraicEllipsoid<DataPoint, _WFunctor, T>::VectorType
AlgebraicEllipsoid<DataPoint, _WFunctor, T>::project( const VectorType& _q ) const
{
    PONCA_MULTIARCH_STD_MATH(abs);

    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    const VectorType grad = primitiveGradient(_q);
    const Scalar a = grad.transpose() * m_uq * grad;
    const Scalar b = grad.squaredNorm();
    const Scalar c = potential(_q);

    // solve a t^2 + b t + c = 0
    Scalar t = 0;

    if(abs(a) < epsilon)
    {
        if(abs(b) < epsilon)
        {
            t = 0;
        }
        else
        {
            t = - c / b;
        }
    }
    else
    {
        const Scalar delta = b*b - 4*a*c;

        if(delta >= 0)
        {
            t = (- b + sqrt(delta)) / (2 * a);
        }
        else
        {
            t = 0; // no solution so no projection
        }
    }

    return _q + t * grad;
}


template < class DataPoint, class _WFunctor, typename T>
typename AlgebraicEllipsoid<DataPoint, _WFunctor, T>::VectorType
AlgebraicEllipsoid<DataPoint, _WFunctor, T>::primitiveGradient( const VectorType &_q ) const
{
        // turn to centered basis
        const VectorType lq = Base::m_w.convertToLocalBasis(_q);
        return m_ul + Scalar(2.f) * m_uq * lq;
}
