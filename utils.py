#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.h5 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import numpy as np
import skimage
from scipy.ndimage import  map_coordinates, zoom
import nibabel as nib


def densewarp(src, warp, interp_order = 0):
    shape = src.shape
    grid = np.mgrid[0:shape[0],0:shape[1],0:shape[2]].astype(float)
    grid[0] += warp[:,:,:,0]
    grid[1] += warp[:,:,:,1]
    grid[2] += warp[:,:,:,2]
    return map_coordinates(src, grid, order = interp_order)

def densewarpPath(src_pth, warp_pth, warped_pth, interp_order):
    src_img = nib.load(src_pth)
    src = src_img.get_fdata()
    warp_img = nib.load(warp_pth)
    warp = warp_img.get_fdata()
    warped = densewarp(src, warp, interp_order=interp_order)
    warped_img = nib.Nifti1Image(warped, src_img.affine)
    nib.save(warped_img, warped_pth)
    return warped_pth


def npDice(v1, v2):
    return np.sum(np.logical_and(v1,v2))*2 / (np.sum(v1)+np.sum(v2))

def labelDice(v1, v2):
    max1 = np.max(v1)
    max2 = np.max(v2)
    max = np.max([max1, max2]).astype(np.int16)
    result = {}
    for i in range (0,max+1):
        result[i] = npDice(v1==i, v2==i)
    return result

def labelDicePath(v1_pth, v2_pth):
    img1 = nib.load(v1_pth)
    img2 = nib.load(v2_pth)
    v1 = img1.get_fdata()
    v2 = img2.get_fdata()
    return labelDice(v1, v2)


def is_affine_orthogonal(affine,tolerance=0.001):
    affine2 = affine[0:3,0:3]
    row_sums = affine2.sum(axis=1)
    new_matrix = affine2 / row_sums[:, np.newaxis]
    return np.sum(new_matrix-np.eye(3)) < tolerance

def is_affine_isotropic(affine,tolerance=0.0001):
    axis = get_affine_axis_ratio(affine)
    m = np.mean(axis)
    return  np.sum(np.float_power(axis - m, 2)) < tolerance

def get_affine_axis_ratio(affine, abs = False):
    if abs:
        return np.abs(np.linalg.eigvals(affine[0:3,0:3]))
    else:
        return np.linalg.eigvals(affine[0:3,0:3])

def resample_to_isotropic(in_array, affine, voxel_size = None, ret_affine=False):
    if not is_affine_orthogonal(affine):
        print("non-orthogonal matrix is not supported..")
        return in_array
    if is_affine_isotropic(affine):
        return in_array
    axis_ratio = get_affine_axis_ratio(affine, abs=True)
    if voxel_size == None or voxel_size<=0:
        min_edge = np.min(axis_ratio)
    else:
        min_edge = voxel_size
    # here we're doing almost alway upscaling, thus area interp is not needed.
    return np.expand_dims(zoom(np.squeeze(in_array),  axis_ratio/min_edge), axis=(0,4))
