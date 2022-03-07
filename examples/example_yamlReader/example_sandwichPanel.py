from dolfin import *
# from dolfin import compile_extension_module
import time
import math
import numpy as np
from petsc4py import PETSc
import os
#import matplotlib.pyplot as plt

from anba4 import *
from ANBAInput import *  ## Experimental yaml reader

# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize_flags"] = "-O2"
# parameters["form_compiler"]["quadrature_degree"] = 2

## Define input file name and read it to construct anba object
inputFileName = 'sandwichPanel.yaml'
anba = readANBAYaml(inputFileName)

# stiff = anba.compute()
# stiff.view()
