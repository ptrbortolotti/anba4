from dolfin import *
#from dolfin import compile_extension_module
import time
import math
import numpy as np
from petsc4py import PETSc
import os
import sys
import matplotlib.pyplot as plt

from anba4 import *

sys.path.append("../../anba4")
from ANBAInput import *

## Define input file name and read it to construct anba object
inputFileName = 'sandwichPanel_simplified_triEls.yaml'
anba = readANBAYaml(inputFileName)

print('created anbax object')

stiff = anba.compute()
stiff.view()

mass = anba.inertia()
mass.view()

anba.stress_field([0., 0., 1.,], [0., 0., 0.], "local", "paraview")
anba.strain_field([0., 0., 1.,], [0., 0., 0.], "local", "paraview")

outputFileName = 'sanwichPanel_output.yaml'
writeResultYaml(anba,outputFileName)

print('computed stiffness')
