import pytest
import numpy as np
import dolfin
from anba4 import *
import mshr


reference_stiffness = """4.6273006070753991e-02 -5.4883197410758094e-06 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0312967608794575e-06 
-5.4883197420143236e-06 6.1096737897345668e-02 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -1.8466986309922783e-02 
0.0000000000000000e+00 0.0000000000000000e+00 3.7999999999939005e-01 5.3442775265196823e-16 -1.0899999999977911e-01 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 1.3301945068637399e-15 2.2926666666658468e-01 -6.6782172517486794e-16 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -1.0899999999977820e-01 -5.5185055152608306e-16 6.7266666666500566e-02 0.0000000000000000e+00 
1.0312967583384075e-06 -1.8466986309932518e-02 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 6.0558452999570772e-03 """

reference_mass = """3.7999999999999989e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 -5.6920614055488983e-18 
0.0000000000000000e+00 3.7999999999999989e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 1.0899999999999982e-01 
0.0000000000000000e+00 0.0000000000000000e+00 3.7999999999999989e-01 5.6920614055488983e-18 -1.0899999999999982e-01 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 5.6920614055488983e-18 2.2926666666666656e-01 3.7947076036992655e-18 0.0000000000000000e+00 
0.0000000000000000e+00 0.0000000000000000e+00 -1.0899999999999982e-01 3.7947076036992655e-18 6.7266666666666558e-02 0.0000000000000000e+00 
-5.6920614055488983e-18 1.0899999999999982e-01 0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00 2.9653333333333382e-01 """


def parse_matrix(ref_str):
    lines = ref_str.strip().split("\n")
    mat = np.array([list(map(float, line.split())) for line in lines if line.strip()])
    return mat


@pytest.fixture(scope="module")
def c_section_data():
    dolfin.parameters["form_compiler"]["optimize"] = True
    dolfin.parameters["form_compiler"]["quadrature_degree"] = 2
    E = 1.0
    nu = 0.33
    matMechanicProp = [E, nu]
    thickness = 0.1
    Square1 = mshr.Rectangle(dolfin.Point(0.0, -1.0, 0.0), dolfin.Point(1.0, 1.0, 0.0))
    Square2 = mshr.Rectangle(
        dolfin.Point(thickness, -1 + thickness, 0),
        dolfin.Point(2.0, 1.0 - thickness, 0),
    )
    C_shape = Square1 - Square2
    mesh = mshr.generate_mesh(C_shape, 64)
    materials = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = dolfin.MeshFunction("double", mesh, mesh.topology().dim())
    materials.set_all(0)
    fiber_orientations.set_all(0.0)
    plane_orientations.set_all(90.0)
    mat1 = material.IsotropicMaterial(matMechanicProp, 1.0)
    matLibrary = [mat1]

    anba = anbax(mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations)
    stiff = anba.compute()
    # stiff.view()

    mass = anba.inertia()
    # mass.view()

    stiff_mat_reg = stiff.getValues(range(6), range(6))

    decoupled_stiff_reg = DecoupleStiffness(stiff_mat_reg)
    mass_mat_reg = mass.getValues(range(6), range(6))

    anba = anbax(
        mesh,
        2,
        matLibrary,
        materials,
        plane_orientations,
        fiber_orientations,
        singular=True,
    )
    stiff_sing = anba.compute()
    # stiff.view()

    mass_sing = anba.inertia()
    # mass.view()

    stiff_mat_sing = stiff_sing.getValues(range(6), range(6))
    decoupled_stiff_sing = utils.DecoupleStiffness(stiff_mat_sing)
    mass_mat_sing = mass_sing.getValues(range(6), range(6))
    # decoupled_mass_sing = utils.DecoupleStiffness(mass_mat_sing)

    stiff_mat_reg = stiff.getValues(range(6), range(6))

    decoupled_stiff_reg = DecoupleStiffness(stiff_mat_reg)
    mass_mat_reg = mass.getValues(range(6), range(6))

    return {
        "stiff_reg": stiff_mat_reg,
        "stiff_sing": stiff_mat_sing,
        "mass_reg": mass_mat_reg,
        "mass_sing": mass_mat_sing,
        "dec_stiff_reg": decoupled_stiff_reg,
        "dec_stiff_sing": decoupled_stiff_sing,
    }


def test_C_section_stiffness_regular_vs_singular(c_section_data):
    np.testing.assert_allclose(
        c_section_data["stiff_reg"], c_section_data["stiff_sing"], atol=1e-6
    )


def test_C_section_mass_regular_vs_singular(c_section_data):
    np.testing.assert_allclose(
        c_section_data["mass_reg"], c_section_data["mass_sing"], atol=1e-6
    )


def test_C_section_dec_stiff_regular_vs_singular(c_section_data):
    np.testing.assert_allclose(
        c_section_data["dec_stiff_reg"], c_section_data["dec_stiff_sing"], atol=1e-6
    )


def test_C_section_stiffness_reference(c_section_data):
    ref_stiff = parse_matrix(reference_stiffness)
    np.testing.assert_allclose(c_section_data["stiff_reg"], ref_stiff, atol=1e-6)


def test_C_section_mass_reference(c_section_data):
    ref_mass = parse_matrix(reference_mass)
    np.testing.assert_allclose(c_section_data["mass_reg"], ref_mass, atol=1e-6)


def test_C_section_dec_stiff_zeros(c_section_data):
    np.testing.assert_allclose(
        c_section_data["dec_stiff_reg"][0:3, 3:6], np.zeros((3, 3)), atol=1e-6
    )
    np.testing.assert_allclose(
        c_section_data["dec_stiff_reg"][3:6, 0:3], np.zeros((3, 3)), atol=1e-6
    )
