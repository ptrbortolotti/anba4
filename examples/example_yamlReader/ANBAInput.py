## By Evan Anderson, 3/7/2022

## Function designed to read in a yaml file defining a cross-sectional model, and construct an anba object from it.
## It may ultimately be worth adding this as a method to the anbax class.

from dolfin import *
# from dolfin import compile_extension_module
import time
import math
import numpy as np
from petsc4py import PETSc
import os
# import matplotlib.pyplot as plt

from anba4 import *

## Utility function to conveniently read various lines of text into lists, selecting certain data types
def str2NumList(inStr,includeFloat=True,includeText=False):
    noCommas = inStr.replace(',',' ')
    noLnEnd = noCommas.replace('\n',' ')
    lnLst = noLnEnd.split(' ')
    numList = []
    for st in lnLst:
        if(len(st) > 0):
            if(st.isnumeric()):
                numList.append(int(st))
            elif(includeFloat):     
                noDec = st.replace('.','')
                noPlus = noDec.replace('+','')
                noMinus = noPlus.replace('-','')
                noE = noMinus.replace('E','')
                noe = noE.replace('e','')
                if(noe.isnumeric()):
                    numList.append(float(st))
                elif(includeText):
                    numList.append(st)
            elif(includeText):
                numList.append(st)
    return numList

def readANBAYaml(fileName):

    ## Open input file
    inFile = open(fileName,'r')

    ## Initialize data sets
    nodeCoord = []
    elConn = []
    elSets = []
    elProps = []
    mats = []

    ## Read in model data from input file
    fileLine = inFile.readline()
    while(fileLine != ''):
        if('nodes:' in fileLine):
            fileLine = inFile.readline()
            bk = 0
            while(bk == 0):
                crd = str2NumList(fileLine,True,False)
                if(len(crd) < 3):
                    bk = 1
                else:
                    nodeCoord.append(crd)
                    fileLine = inFile.readline()
        elif('elem_connectivity:' in fileLine):
            fileLine = inFile.readline()
            bk = 0
            while(bk == 0):
                elC = str2NumList(fileLine,False,False)
                if(len(elC) < 4):
                    bk = 1
                else:
                    elConn.append(elC)
                    fileLine = inFile.readline()
        elif('elem_set:' in fileLine):
            fileLine = inFile.readline()
            bk = 0
            setLabels = []
            while(bk == 0):
                if('name:' in fileLine):
                    lnLst = fileLine.split(':')
                    setName = lnLst[1].strip()
                    fileLine = inFile.readline()
                elif('elem_labels:' in fileLine):
                    labList = str2NumList(fileLine,False,False)
                    setLabels.extend(labList)
                    fileLine = inFile.readline()
                    bk2 = 0
                    while(bk2 == 0):
                        labList = str2NumList(fileLine,False,False)
                        if(len(labList) > 0):
                            setLabels.extend(labList)
                            fileLine = inFile.readline()
                        else:
                            bk2 = 1
                elif(fileLine.isspace()):
                    fileLine = inFile.readline()
                else:
                    bk = 1
            thisSet = []
            thisSet.append(setName)
            thisSet.extend(setLabels)
            elSets.append(thisSet)
        elif('elem_properties:' in fileLine):
            fileLine = inFile.readline()
            bk = 0
            while(bk == 0):
                numList = str2NumList(fileLine,True,True)
                if(len(numList) == 4):
                    elProps.append(numList)
                    fileLine = inFile.readline()
                else:
                    bk = 1
        elif('material:' in fileLine):
            matRho = 0
            fileLine = inFile.readline()
            bk = 0
            while(bk == 0):
                if('name:' in fileLine):
                    lnLst = fileLine.split(':')
                    matName = lnLst[1].strip()
                    fileLine = inFile.readline()
                elif('type:' in fileLine):
                    lnLst = fileLine.split(':')
                    matType = lnLst[1].strip()
                    fileLine = inFile.readline()
                elif('ex:' in fileLine):
                    lnLst = fileLine.split(':')
                    matEx = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('ey:' in fileLine):
                    lnLst = fileLine.split(':')
                    matEy = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('ez:' in fileLine):
                    lnLst = fileLine.split(':')
                    matEz = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('nuxy:' in fileLine):
                    lnLst = fileLine.split(':')
                    matNuxy = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('nuxz:' in fileLine):
                    lnLst = fileLine.split(':')
                    matNuxz = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('nuyz:' in fileLine):
                    lnLst = fileLine.split(':')
                    matNuyz = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('gxy:' in fileLine):
                    lnLst = fileLine.split(':')
                    matGxy = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('gxz:' in fileLine):
                    lnLst = fileLine.split(':')
                    matGxz = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('gyz:' in fileLine):
                    lnLst = fileLine.split(':')
                    matGyz = float(lnLst[1])
                    fileLine = inFile.readline()
                elif(' e:' in fileLine):
                    lnLst = fileLine.split(':')
                    matEx = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('nu:' in fileLine):
                    lnLst = fileLine.split(':')
                    matNuxy = float(lnLst[1])
                    fileLine = inFile.readline()
                elif('rho:' in fileLine):
                    lnLst = fileLine.split(':')
                    matRho = float(lnLst[1])
                    fileLine = inFile.readline()
                elif(fileLine.isspace() or 'properties:' in fileLine):
                    fileLine = inFile.readline()
                else:
                    bk = 1
            if(matType == 'isotropic'):
                matProps = [matEx,matNuxy]
            else:
                matProps = [matEx,matEy,matEz,matGyz,matGxz,matGxy,matNuyz,matNuxz,matNuxy]
            thisMat = [matName,matType,matProps,matRho]
            mats.append(thisMat)
        else:
            fileLine = inFile.readline()

    inFile.close()

#    print(nodeCoord)
#    print(elConn)
#    print(elSets)
#    print(elProps)
#    print(mats)

    ## Build a mesh object from file data
    mesh = Mesh()
    me = MeshEditor()
    me.open(mesh, "quadrilateral", 2, 2)

    me.init_vertices(len(nodeCoord))
    for nd in nodeCoord:
        me.add_vertex(nd[0]-1, (nd[1] , nd[2]))

    me.init_cells(len(elConn))
    for el in elConn:
        me.add_cell(el[0]-1, [el[1]-1,el[2]-1,el[4]-1,el[3]-1])
        
    me.close()

    ## Build the materials, fiber orientations and plane orientations for each element

    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    fiber_orientations = MeshFunction("double", mesh, mesh.topology().dim())
    plane_orientations = MeshFunction("double", mesh, mesh.topology().dim())

    for ep in elProps:
        matInd = 0
        i = 0
        for mat in mats:
            if(mat[0] == ep[1]):
                matInt = i
            i = i + 1
        elStr = str(ep[0])
        if(elStr.isnumeric()):
            materials.set_value(ep[0]-1,matInd)
            fiber_orientations.set_value(ep[0]-1,ep[2])
            plane_orientations.set_value(ep[0]-1,ep[3])
        else:
            for es in elSets:
                if(es[0] == ep[0]):
                    for i in range(1,len(es)):
                        materials.set_value(es[i]-1,matInd)
                        fiber_orientations.set_value(es[i]-1,ep[2])
                        plane_orientations.set_value(es[i]-1,ep[3])

    ## Build the material library
    matLibrary = []
    for mat in mats:
        if(mat[1] == 'isotropic'):
            newMat = material.IsotropicMaterial(mat[2], mat[3])
            matLibrary.append(newMat)
        elif(mat[1] == 'orthotropic'):
            mp = mat[2]
            matProps = np.zeros((3,3))
            matProps[0,0] = mp[0]
            matProps[0,1] = mp[1]
            matProps[0,2] = mp[2]
            matProps[1,0] = mp[3]
            matProps[1,1] = mp[4]
            matProps[1,2] = mp[5]
            matProps[2,0] = mp[6]
            matProps[2,1] = mp[7]
            matProps[2,2] = mp[8]
            newMat = material.OrthotropicMaterial(matProps, mat[3])
            matLibrary.append(newMat)
        else:
            print('Warning: material type not recognized:')
            print(mat[1])

    ## Create the anbax object from the mesh, materials, and orientation information and return it.
    anba = anbax(mesh, 2, matLibrary, materials, plane_orientations, fiber_orientations)
    return anba
