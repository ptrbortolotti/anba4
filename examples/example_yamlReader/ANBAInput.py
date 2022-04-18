## By Evan Anderson, 3/7/2022

## Function designed to read in a yaml file defining a cross-sectional model, and construct an anba object from it.
## It may ultimately be worth adding this as a method to the anbax class.

from dolfin import *
#from dolfin import compile_extension_module
import time
import math
import numpy as np
from petsc4py import PETSc
import os
from ruamel.yaml import YAML
import matplotlib.pyplot as plt

from anba4 import *

# from validation import validate_with_defaults

# input_data = validate_with_defaults('sandwichPanel.yaml', 'anba4_schema.yaml')


def readANBAYaml(fileName):

    ## Read File
    inFile = open(fileName,'r')
    yamlReader = YAML()
    inpObj = yamlReader.load(inFile)
    inFile.close()

    ## Rename key data members
    nodeCoord = inpObj['nodes']
    elConn = inpObj['elem_connectivity']
    elSets = inpObj['elem_set']
    mats = inpObj['material']
    
    ## Plot input geometry
    numNds = len(nodeCoord)
    pltX = np.zeros((numNds))
    pltY = np.zeros((numNds))
    for nd in nodeCoord:
        lb = nd[0] - 1
        pltX[lb] = nd[1]
        pltY[lb] = nd[2]

    numEls = len(elConn)
    triList = np.zeros((numEls,3))
    for el in elConn:
        lb = el[0] - 1
        triList[lb,0] = el[1] - 1
        triList[lb,1] = el[2] - 1
        triList[lb,2] = el[3] - 1
    
    colors = np.zeros((numEls))
    for i in range(0,len(elSets)):
        for j in elSets[i]['elem_labels']:
            colors[j-1] = i
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.tripcolor(pltX,pltY,triList,facecolors=colors,edgecolors='k')
    
    plt.show()

    ## Build a mesh object from file data
    mesh = Mesh()
    me = MeshEditor()
    me.open(mesh, "triangle", 2, 2)

    me.init_vertices(len(nodeCoord))
    for nd in nodeCoord:
        me.add_vertex(nd[0]-1, np.array([nd[1], nd[2]]))

    me.init_cells(len(elConn))
    for el in elConn:
        me.add_cell(el[0]-1, np.array([el[1]-1,el[2]-1,el[3]-1], dtype=np.uintp))
        
    me.close()

    ## Build the materials, fiber orientations and plane orientations for each element

    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())

    for es in elSets:
        fibAng = es['fiber_angle']
        planeAng = es['plane_angle']
        matInd = 0
        i = 0
        for mat in mats:
            if(mat['name'] == es['material']):
                matInd = i;
            i = i + 1
        eLabs = es['elem_labels']
        for el in eLabs:
            materials.set_value(el-1,matInd)
            fiber_orientations.set_value(el-1,fibAng)
            plane_orientations.set_value(el-1,planeAng)

    ## Build the material library
    matLibrary = []
    for mat in mats:
        if(mat['orth'] == 0):
            newMat = material.IsotropicMaterial([mat['E'],mat['nu']])
            matLibrary.append(newMat)
        else:
            matProps = np.zeros((3,3))
            matProps[0,0] = mat['E'][0]
            matProps[0,1] = mat['E'][1]
            matProps[0,2] = mat['E'][2]
            matProps[1,0] = mat['G'][2]  # Gyz
            matProps[1,1] = mat['G'][1]  # Gxz
            matProps[1,2] = mat['G'][0]  # Gxy
            matProps[2,0] = mat['nu'][2]
            matProps[2,1] = mat['nu'][1]
            matProps[2,2] = mat['nu'][0]
            newMat = material.OrthotropicMaterial(matProps, mat['rho'])
            matLibrary.append(newMat)

    ## Create the anbax object from the mesh, materials, and orientation information and return it.
    anba = anbax(mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations)
    return anba
