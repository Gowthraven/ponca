#include <Eigen/Geometry>

template < class DataPoint, class _WFunctor, typename T>
void
ParabolicCylinder<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    m_ul.setZero();
    m_uq.setZero();
    m_a = Scalar(0);
    m_uc = Scalar(0);

    m_planeIsReady = false;
}

template < class DataPoint, class _WFunctor, typename T>
bool
ParabolicCylinder<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
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
        VectorType local = Base::worldToTangentPlane(attributes.pos());
        const Scalar& f = *(local.data());
        const Scalar& x = *(local.data()+1);
        const Scalar& y = *(local.data()+2);

        Scalar xy = x*y;
        Scalar xx = x*x;
        Scalar yy = y*y;
        
        Eigen::Vector<Scalar, 7> v {1, x, y , xx, yy, xy, xy};

        m_A_cov += v * v.transpose() * w;
        m_F_cov += f * w * v;

        return true;
    }
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::potential( const VectorType &_q ) const
{  
    VectorType x = Base::worldToTangentPlane(_q);
    return eval_quadratic_function(*(x.data() +1 ), *(x.data() + 2));
}


template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::project( const VectorType& _q ) const
{   
    VectorType x = Base::worldToTangentPlane(_q);
    return Base::project(_q) + (eval_quadratic_function(*(x.data()+1), *(x.data()+2)) * Base::primitiveGradient(_q));
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::plane_project( const VectorType& _q ) const
{   
    return Base::project(_q);
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::primitiveGradient( const VectorType& _q ) const
{
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
ParabolicCylinder<DataPoint, _WFunctor, T>::finalize () {

    if (! m_planeIsReady) {
        FIT_RESULT res = Base::finalize();

        if (res == STABLE) {
            m_planeIsReady = true;
            m_A_cov = SampleMatrix (7, 7);
            m_F_cov = SampleVector (7);
            m_A_cov.setZero();
            m_F_cov.setZero();

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
ParabolicCylinder<DataPoint, _WFunctor, T>::all_minimisation () {
    first_minimisation();
    second_minimisation();
    third_minimisation();
    return 1;
}

template < class DataPoint, class _WFunctor, typename T>
int
ParabolicCylinder<DataPoint, _WFunctor, T>::first_minimisation () {
    PONCA_MULTIARCH_STD_MATH(abs);
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    Eigen::BDCSVD<Eigen::MatrixXd> svd(m_A_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Vector7 x = svd.solve(m_F_cov);

    m_uc = x(0,0);     // Used to make test, you may just run the first minimisation to see a 2D ellipsoid. (just comment the line in eval quadratic function and in the primitive gradient)
    m_ul(0) = x(1,0);  // Used to make test also.
    m_ul(1) = x(2,0);  // Used to make test also.
    m_H(0,0) = x(3,0);
    m_H(1,1) = x(4,0);
    m_H(0,1) = x(5,0);
    m_H(1,0) = x(6,0);

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

    return 1;
}

template < class DataPoint, class _WFunctor, typename T>
int
ParabolicCylinder<DataPoint, _WFunctor, T>::second_minimisation () { 

    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();
    
    Scalar u0_squared = m_uq(0) * m_uq(0);
    if (u0_squared < epsilon) u0_squared = 0;
    Scalar u1_squared = m_uq(1) * m_uq(1);
    if (u1_squared < epsilon) u1_squared = 0;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4, 4);
    A.block (0, 0, 3, 3) = m_A_cov.block (0, 0, 3, 3);
    A(0, 3) = u0_squared * m_A_cov(0, 3) + 2 * m_uq(0) * m_uq(1) * m_A_cov(0,5) + u1_squared * m_A_cov(0, 4);
    A(1, 3) = u0_squared * m_A_cov(1, 3) + 2 * m_uq(0) * m_uq(1) * m_A_cov(1,5) + u1_squared * m_A_cov(1, 4);
    A(2, 3) = u0_squared * m_A_cov(2, 3) + 2 * m_uq(0) * m_uq(1) * m_A_cov(2,5) + u1_squared * m_A_cov(2, 4);
    A(3, 3) = u0_squared * u0_squared * m_A_cov(3, 3) + 
                u1_squared * u1_squared * m_A_cov(4, 4) +
                    6 * ( u0_squared * u1_squared * m_A_cov(3, 4) ) +
                        4 * ( u0_squared * m_uq(0) * m_uq(1) * m_A_cov(3, 5) ) +
                            4 * ( m_uq(0) * u1_squared * m_uq(1) * m_A_cov(4, 5) );
    A(3, 0) = A(0, 3);
    A(3, 1) = A(1, 3);
    A(3, 2) = A(2, 3);

    Eigen::VectorXd F = Eigen::VectorXd::Zero(4);
    F(0) = m_F_cov(0);
    F(1) = m_F_cov(1);
    F(2) = m_F_cov(2);
    F(3) = u0_squared * m_F_cov(3) + 2 * m_uq(0) * m_uq(1) * m_F_cov(5) + u1_squared * m_F_cov(4);

    Eigen::Matrix<Scalar, 4, 1> x = (A).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(F);
    m_a *= x(3,0);

    return 1;

}

template < class DataPoint, class _WFunctor, typename T>
void
ParabolicCylinder<DataPoint, _WFunctor, T>::third_minimisation () {
    Scalar u0_squared = m_uq(0) * m_uq(0);
    Scalar u1_squared = m_uq(1) * m_uq(1);

    Scalar val_1 = m_a * (u0_squared * m_A_cov(0, 3) + 2 * m_uq(0) * m_uq(1) * m_A_cov(0, 5) + u1_squared * m_A_cov(0, 4));
    Scalar val_x = m_a * (u0_squared * m_A_cov(1, 3) + 2 * m_uq(0) * m_uq(1) * m_A_cov(1, 5) + u1_squared * m_A_cov(1, 4));
    Scalar val_y = m_a * (u0_squared * m_A_cov(2, 3) + 2 * m_uq(0) * m_uq(1) * m_A_cov(2, 5) + u1_squared * m_A_cov(2, 4));

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    A.block (0, 0, 3, 3) = m_A_cov.block (0, 0, 3, 3);

    Eigen::VectorXd F = Eigen::VectorXd::Zero(3);
    F(0) = m_F_cov(0) - val_1;
    F(1) = m_F_cov(1) - val_x;
    F(2) = m_F_cov(2) - val_y;

    Eigen::Matrix<Scalar, 3, 1> x = (A).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(F);

    m_uc = x(0,0);
    m_ul(0) = x(1,0);
    m_ul(1) = x(2,0);
}
