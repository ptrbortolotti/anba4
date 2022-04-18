from dolfin import *
#from dolfin import compile_extension_module
import time
import math
import numpy as np
from petsc4py import PETSc
import os
import matplotlib.pyplot as plt

from anba4 import *
from ANBAInput import *  ## Experimental yaml reader

## Define input file name and read it to construct anba object
inputFileName = 'sandwichPanel_simplified_triEls.yaml'
anba = readANBAYaml(inputFileName)

print('created anbax object')

stiff = anba.compute()
stiff.view()

print('computed stiffness')
