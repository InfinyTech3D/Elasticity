"""Tet-element entry point for `runSofa`, mirrors sinusoidal.py but loads
the P1 tetrahedron scene (element_tet) instead of the default Q1 hex.

Usage:  runSofa sinusoidal_tet.py
"""

from sinusoidal import mms
from solid import case_scene, run_reference_scene, element_tet

createScene = case_scene(mms, element_tet)

if __name__ == "__main__":
    run_reference_scene(element_tet, mms)