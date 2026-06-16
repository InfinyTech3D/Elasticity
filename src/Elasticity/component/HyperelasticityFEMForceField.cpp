#define ELASTICITY_COMPONENT_HYPERLASTICITY_FEM_FORCE_FIELD_CPP

#include <Elasticity/component/HyperelasticityFEMForceField.inl>

#include <sofa/fem/FiniteElement[all].h>
#include <sofa/core/ObjectFactory.h>

namespace elasticity
{

void registerHyperelasticityFEMForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Hyperelasticity")
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron> >()
    );
}

template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class ELASTICITY_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

}
