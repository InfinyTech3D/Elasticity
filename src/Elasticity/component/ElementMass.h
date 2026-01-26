#pragma once

#include <Elasticity/config.h>
#include <sofa/core/behavior/Mass.h>

#if !defined(ELASTICITY_COMPONENTS_ELEMENT_MASS_CPP)
#include <Elasticity/finiteelement/FiniteElement[all].h>
#endif

namespace elasticity
{

template <class DataTypes, class ElementType>
class ElementMass : public sofa::core::behavior::Mass<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ElementMass, DataTypes, ElementType),
           sofa::core::behavior::Mass<DataTypes>);

    /**
     * The purpose of this function is to register the name of this class according to the provided
     * pattern.
     *
     * Example: ElementMass<Vec3Types, sofa::geometry::Edge> will produce
     * the class name "EdgeMass".
     */
    static const std::string GetCustomClassName()
    {
        return std::string(sofa::geometry::elementTypeToString(ElementType::Element_type)) +
               "Mass";
    }

    static const std::string GetCustomTemplateName() { return DataTypes::Name(); }

    bool isDiagonal() const override;

    void addForce(
        const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& f,
        const sofa::DataVecCoord_t<DataTypes>& x,
        const sofa::DataVecDeriv_t<DataTypes>& v) override;

};



#if !defined(ELASTICITY_COMPONENTS_ELEMENT_MASS_CPP)
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class ELASTICITY_API ElementMass<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}
