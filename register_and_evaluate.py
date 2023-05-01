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
import voxelmorph as vxm
import tensorflow as tf
import skimage
from scipy.ndimage import gaussian_filter, zoom

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

def resample_to_isotropic(in_array, affine, mode="smallest", ret_affine=False):
    if not is_affine_orthogonal(affine):
        print("non-orthogonal matrix is not supported..")
        return in_array
    if is_affine_isotropic(affine):
        return in_array
    axis_ratio = get_affine_axis_ratio(affine, abs=True)
    min_edge = np.min(axis_ratio)
    # here we're doing almost alway upscaling, thus area interp is not needed.
    return zoom(in_array, axis_ratio/min_edge )

    

def register128(moving_pth, fixed_pth, moved_pth, model, warp_pth=None, gpu=None, 
                multichannel=False, method="slidingwindow", window_size = np.array([128,128,128]),
                overlap = np.array([16,16,16]), gaussian_scale=1./8, sub_pth=None, sub_pth_moved=None,
                mask_border = np.array([0,0,0])):
    '''perform register and return path to warp and moved images'''
    # tensorflow device handling
    device, nb_devices = vxm.tf.utils.setup_device(gpu)

    # load moving and fixed images
    add_feat_axis = not multichannel
    moving, moving_affine = vxm.py.utils.load_volfile(
        moving_pth, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        fixed_pth, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

    

    inshape = moving.shape[1:-1]
    nb_feats = moving.shape[-1] #channel

    orig_moving = np.copy(moving)
    orig_fixed  = np.copy(fixed)
    #normalization.
    moving = (moving - np.min(moving))/(np.max(moving) - np.min(moving))
    fixed  = (fixed  - np.min(fixed ))/(np.max(fixed ) - np.min(fixed ))

    # gaussian
    if inshape[0]>128 or inshape[1]>128 or inshape[2]>128:
        if method == "slidingwindow":
            # Method1: slice image to obtain 128x128x128 chunks. 
            masked_window_size = window_size - mask_border*2
            mask_slice = (slice(None),slice(mask_border[0], window_size[0] - mask_border[0]),
                                      slice(mask_border[1], window_size[1] - mask_border[1]),
                                      slice(mask_border[2], window_size[2] - mask_border[2]),slice(None))
            window_shift = masked_window_size -  overlap
            window_num = np.ceil((inshape - overlap) / window_shift)
            padded_size = (window_num * window_shift + overlap).astype(np.int16)
            padded_fixed =  np.pad(fixed,  [[0,0],[0, padded_size[0]-inshape[0]], [0, padded_size[1]-inshape[1]],[0,padded_size[2]-inshape[2]],[0,0]])
            padded_moving = np.pad(moving, [[0,0],[0, padded_size[0]-inshape[0]], [0, padded_size[1]-inshape[1]],[0,padded_size[2]-inshape[2]],[0,0]])
            padded_nb_sample = np.zeros_like(padded_moving)
            #generate gaussian map
            tmp = np.zeros(window_size)
            tmp [ [i // 2 for i in window_size]] = 1
            gaussian_map = np.expand_dims(gaussian_filter(tmp, sigma = gaussian_scale*window_size), axis=[0,4])
            gaussian_map3 = np.concatenate([gaussian_map, gaussian_map, gaussian_map], axis=4)
            xx, yy, zz = np.mgrid[0:window_num[0], 0:window_num[1], 0:window_num[2]]
            grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.astype(np.int16)
            grid_coordinates = [ [(window_shift[0]*x),  overlap[0]+window_shift[0]*(x+1), 
                                  (window_shift[1]*y),  overlap[1]+window_shift[1]*(y+1),
                                  (window_shift[2]*z),  overlap[2]+window_shift[2]*(z+1)] for x,y,z in grid]
            grid_slices = [(slice(None),slice(g[0],g[1]),slice(g[2],g[3]),slice(g[4],g[5]),slice(None)) for g in grid_coordinates]
            infer_slices = [(slice(None),slice(g[0],g[1]),slice(g[2],g[3]),slice(g[4],g[5]),slice(None)) for g in grid_coordinates]
            grid_count = len(grid_coordinates)
            padded_warp =np.zeros([padded_fixed.shape[0],padded_fixed.shape[1],padded_fixed.shape[2],padded_fixed.shape[3],3])
            config = dict(inshape=window_size, input_model=None)
            tf.config.run_functions_eagerly(True)
            with tf.device(device):
                print("loading synmorph model...")
                synthmorph = vxm.networks.VxmDense.load(model, **config)
                for i, grid_slice in enumerate(grid_slices):
                    # load model and predict
                    print("Registering grid#{}/{}".format(i+1, grid_count))
                    warp_patch = synthmorph.register(padded_moving[grid_slice], padded_fixed[grid_slice])
                    padded_warp[grid_slice] += warp_patch[mask_slice] * gaussian_map3[mask_slice]
                    padded_nb_sample[grid_slice] += gaussian_map[mask_slice]
            padded_warp = padded_warp/ np.concatenate([padded_nb_sample, padded_nb_sample, padded_nb_sample], axis=4)
            warp = padded_warp[:,0:inshape[0],0:inshape[1],0:inshape[2],:]
            vxm.py.utils.save_volfile(warp.squeeze(), warp_pth, fixed_affine)

            print("done registration, now perform transform")
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([orig_moving, warp])
        elif method == "shrink":
            # Method2: shirnk images to fit 128x128x128, calculate warp, then re-expand warp to fit original size, then performe warping.
            dim_max = np.max(inshape)
            re_size = np.ceil((inshape/dim_max)*128).astype(np.int16)
            re_fixed  = skimage.transform.resize_local_mean(fixed,  [1, re_size[0], re_size[1], re_size[2],1])
            re_moving = skimage.transform.resize_local_mean(moving, [1, re_size[0], re_size[1], re_size[2],1])
            re_fixed = np.pad(re_fixed, [[0,0],[0,128-re_size[0]],[0,128-re_size[1]],[0,128-re_size[2]],[0,0]])
            re_moving = np.pad(re_moving, [[0,0],[0,128-re_size[0]],[0,128-re_size[1]],[0,128-re_size[2]],[0,0]])
            with tf.device(device):
                config = dict(inshape=[128,128,128], input_model=None)
                synthmorph = vxm.networks.VxmDense.load(model, **config)
                re_warp = synthmorph.register(re_moving, re_fixed)
                warp = skimage.transform.resize_local_mean(re_warp, [1, dim_max, dim_max, dim_max,3])*(dim_max/128)  # displacement also increased after upsizing.
                warp = warp[:,0:inshape[0],0:inshape[1],0:inshape[2],:]
                print("done registration, now perform transform")
                moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([orig_moving, warp])
    else:
        with tf.device(device):
            # load model and predict
            config = dict(inshape=inshape, input_model=None)
            warp = vxm.networks.VxmDense.load(model, **config).register(moving, fixed)
            print("done registration, now perform transform")
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([orig_moving, warp])


    print("done transform, save images.")
    # save warp
    if warp_pth:
        vxm.py.utils.save_volfile(warp.squeeze(), warp_pth, fixed_affine)

    # save moved image
    vxm.py.utils.save_volfile(moved.squeeze(), moved_pth, fixed_affine)

    if sub_pth:
        sub = orig_moving - orig_fixed
        vxm.py.utils.save_volfile(sub.squeeze(), sub_pth, fixed_affine)
    
    if sub_pth_moved:
        sub_moved = moved -  orig_fixed
        vxm.py.utils.save_volfile(sub_moved.squeeze(), sub_pth_moved, fixed_affine)

moving_pth = "data/C+_filled.nii.gz"
fixed_pth  = "data/C-_filled.nii.gz"
moved_pth  = "data/C+_filled_sl_moved.nii.gz"
warp_pth   = "data/C+_filled_sl_warp.nii.gz"
model      = "models/02000.h5"
sub_pth    = "data/C+_sub.nii.gz"
sub_pth_moved = "data/C+_sub_moved.nii.gz"
register128(moving_pth, fixed_pth, moved_pth, model, warp_pth=warp_pth, gpu=None, 
                multichannel=False, overlap = np.array([32,32,32]),
                method="slidingwindow", sub_pth=sub_pth, sub_pth_moved=sub_pth_moved, mask_border=np.array([16,16,16]))
