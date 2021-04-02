import numpy as np
import SimpleITK as sitk

def getIJKToRasFromVolume(vol):
    trans = vtk.vtkMatrix4x4()
    vol.GetIJKToRASDirectionMatrix(trans)
    return matToArr(trans)

def matToArr(mat):
    return np.array([[mat.GetElement(j, i) for i in range(4)] for j in range(4)])

def arrToMat(arr):
    if arr.shape==(4, 4):
        trans = vtk.vtkMatrix4x4()
    elif arr.shape==(3, 3):
        trans = vtk.vtkMatrix3x3()
    else:
        raise ValueError(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            trans.SetElement(i, j, arr[i, j])
    return trans

def getNewVolAndTransform(curVol, newName):
    acqMatrix = getIJKToRasFromVolume(curVol) # W, H, D / RAS
    spacing = np.array(curVol.GetSpacing())
    origin = np.array(curVol.GetOrigin())
    volArr = arrayFromVolume(curVol) # D, H, W
    principleElementIdx = np.argmax(np.abs(acqMatrix), axis=0)[:3]
    principleElement = np.sign(acqMatrix[principleElementIdx, np.arange(3)])[principleElementIdx.argsort()]
    targetPrincipleElement = np.array([1 ,1, 1])
    targetPrincipleElement[principleElement != [1, 1, 1]] = -1
    volArrNew = volArr.transpose(principleElementIdx[::-1].argsort()[::-1])
    acqMatrixNew = acqMatrix[:, principleElementIdx.argsort().tolist() + [3]]
    spacingNew = spacing[principleElementIdx.argsort()]
    originIdxAfterReverse = np.array([0, 0, 0])
    volArrNew = volArrNew[::int(targetPrincipleElement[2]), ::int(targetPrincipleElement[1]), ::int(targetPrincipleElement[0])]
    
    for i in range(3):
        if targetPrincipleElement[i] < 0:
            # reverse
            originIdxAfterReverse[principleElementIdx.argsort()[i]] = volArrNew.shape[2-i] - 1
    
    for i in range(3):
        if targetPrincipleElement[i] < 0:
            # reverse
            acqMatrixNew[:, i] = - acqMatrixNew[:, i]
    
    originNew = np.matmul(acqMatrix[:3, :3], spacing * originIdxAfterReverse) + origin
    volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    volumeNode.SetName(newName + '_transformed')
    volumeNode.CreateDefaultDisplayNodes()
    updateVolumeFromArray(volumeNode, volArrNew)
    volumeNode.SetSpacing(spacingNew.tolist())
    transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode')
    transformNode.SetName(newName + '_acqMatrix')
    acqMatrixNew[:3, 3]  = originNew
    transformNode.SetMatrixTransformToParent(arrToMat(acqMatrixNew))
    setSliceViewerLayers(background=volumeNode, fit=True)

allVols = [i for i in getNodes() if 'vtkMRMLScalarVolumeNode' in str(type(getNode(i)))]
pendingVols = [i for i in allVols if not '_transformed' in i and not i + "_transformed" in allVols]
for k in pendingVols:
    getNewVolAndTransform(getNode(k), k)


