#pragma once

#include <Elasticity/config.h>
#include <Elasticity/impl/trait.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/behavior/TopologyAccessor.h>

#if !defined(ELASTICITY_COMPONENTS_ELEMENT_MASS_CPP)
#include <Elasticity/finiteelement/FiniteElement[all].h>
#endif

namespace elasticity
{

template <class DataTypes, class ElementType>
class ElementMass :
    public sofa::core::behavior::Mass<DataTypes>,
    public sofa::core::behavior::TopologyAccessor
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(ElementMass, DataTypes, ElementType),
           sofa::core::behavior::Mass<DataTypes>,
           sofa::core::behavior::TopologyAccessor);

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

    void init() override;

    void addForce(
        const sofa::core::MechanicalParams* mparams,
        sofa::DataVecDeriv_t<DataTypes>& f,
        const sofa::DataVecCoord_t<DataTypes>& x,
        const sofa::DataVecDeriv_t<DataTypes>& v) override;

    sofa::Data<sofa::VecReal_t<DataTypes> > d_nodalDensity;

protected:

    using trait = elasticity::trait<DataTypes, ElementType>;
    using FiniteElement = typename trait::FiniteElement;

    ElementMass();

    static constexpr sofa::Real_t<DataTypes> defaultNodalDensity { 1. };
    void resizeNodalDensity(const std::size_t size);

    sofa::Deriv_t<DataTypes> getGravity() const;

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
