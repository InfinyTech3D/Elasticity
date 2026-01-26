#pragma once
#include <Elasticity/component/ElementMass.h>

namespace elasticity
{

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
}

}  // namespace elasticity
