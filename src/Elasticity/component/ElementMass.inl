#pragma once
#include <Elasticity/component/ElementMass.h>
#include <Elasticity/impl/VectorTools.h>
#include <Elasticity/impl/MatrixTools.h>

namespace elasticity
{

template <class DataTypes, class ElementType>
ElementMass<DataTypes, ElementType>::ElementMass()
    : d_nodalDensity(initData(&d_nodalDensity, {defaultNodalDensity}, "nodalDensity", "Scalar density assigned to each node. The density is interpolated between nodes using the element shape functions."))
{
}

template <class DataTypes, class ElementType>
void ElementMass<DataTypes, ElementType>::init()
{
    sofa::core::behavior::Mass<DataTypes>::init();

    if (!this->isComponentStateInvalid())
    {
        sofa::core::behavior::TopologyAccessor::init();
    }

    if (!this->isComponentStateInvalid() && this->mstate)
    {
        this->resizeNodalDensity(this->mstate->getSize());
    }

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

template <class DataTypes, class ElementType>
void ElementMass<DataTypes, ElementType>::resizeNodalDensity(const std::size_t size)
{
    sofa::helper::WriteAccessor nodalDensity = sofa::helper::getWriteAccessor( d_nodalDensity);

    if (nodalDensity.size() < size)
    {
        if (nodalDensity.empty())
        {
            nodalDensity.resize(size, defaultNodalDensity);
        }
        else
        {
            nodalDensity.resize(size, nodalDensity.back());
        }
    }
}

template <class DataTypes, class ElementType>
sofa::Deriv_t<DataTypes> ElementMass<DataTypes, ElementType>::getGravity() const
{
    const auto& nodeGravity = this->getContext()->getGravity();

    sofa::Deriv_t<DataTypes> gravity;
    DataTypes::set ( gravity, nodeGravity[0], nodeGravity[1], nodeGravity[2]);

    return gravity;
}

template <class DataTypes, class ElementType>
bool ElementMass<DataTypes, ElementType>::isDiagonal() const
{
    return false;
}

template <class DataTypes, class ElementType>
void ElementMass<DataTypes, ElementType>::addForce(const sofa::core::MechanicalParams* mparams,
                                                   sofa::DataVecDeriv_t<DataTypes>& f,
                                                   const sofa::DataVecCoord_t<DataTypes>& x,
                                                   const sofa::DataVecDeriv_t<DataTypes>& v)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(v);

    if (!l_topology)
        return;

    const sofa::helper::ReadAccessor nodalDensity = sofa::helper::getReadAccessor( d_nodalDensity);

    auto positionAccessor = sofa::helper::getReadAccessor(x);
    auto forceAccessor = sofa::helper::getWriteAccessor(f);

    //make sure there is a density associated to each node
    resizeNodalDensity(positionAccessor.size());

    const auto gravity = this->getGravity();

    const auto& elements = FiniteElement::getElementSequence(*this->l_topology);

    for (std::size_t elementId = 0; elementId < elements.size(); ++elementId)
    {
        const auto& element = elements[elementId];

        const std::array<sofa::Coord_t<DataTypes>, trait::NumberOfNodesInElement> elementNodesCoordinates =
            extractNodesVectorFromGlobalVector(element, positionAccessor.ref());

        const std::array<sofa::Real_t<DataTypes>, trait::NumberOfNodesInElement> elementNodesDensity =
            extractNodesVectorFromGlobalVector(element, nodalDensity.ref());

        // Integrate: f_i += ∫_Ω rho(x) * N_i(x) * g dΩ
        // with rho(x) interpolated from nodal densities: rho(x) = Σ_j N_j(x) * rho_j
        for (const auto& [quadraturePoint, weight] : FiniteElement::quadraturePoints())
        {
            // shape functions evaluated at the quadrature point
            const auto N = FiniteElement::shapeFunctions(quadraturePoint);

            // gradient of shape functions in the reference element evaluated at the quadrature point
            const auto dN_dq_ref = FiniteElement::gradientShapeFunctions(quadraturePoint);

            // jacobian of the mapping from the reference space to the physical space
            sofa::type::Mat<DataTypes::spatial_dimensions, FiniteElement::TopologicalDimension, sofa::Real_t<DataTypes>> jacobian;
            for (sofa::Size i = 0; i < trait::NumberOfNodesInElement; ++i)
            {
                jacobian += sofa::type::dyad(elementNodesCoordinates[i], dN_dq_ref[i]);
            }

            const auto detJ = elasticity::absGeneralizedDeterminant(jacobian);
            const auto dV = weight * detJ;

            // density at quadrature point (interpolated)
            sofa::Real_t<DataTypes> rho_q = static_cast<sofa::Real_t<DataTypes>>(0);
            for (sofa::Size j = 0; j < trait::NumberOfNodesInElement; ++j)
            {
                rho_q += N[j] * elementNodesDensity[j];
            }

            // distribute body force to element nodes
            for (sofa::Size i = 0; i < trait::NumberOfNodesInElement; ++i)
            {
                forceAccessor[element[i]] += (dV * (rho_q * N[i])) * gravity;
            }
        }
    }
}

}  // namespace elasticity
