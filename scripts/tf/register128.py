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

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import skimage


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--slidingwindow', action='store_true',
                    help='use slidingwindow for inference')
parser.add_argument('--shrink', action='store_true',
                    help='use resized image for inference')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

inshape = moving.shape[1:-1]
nb_feats = moving.shape[-1] #channel

orig_moving = np.copy(moving)
#normalization.
moving = (moving - np.min(moving))/(np.max(moving) - np.min(moving))
fixed  = (fixed  - np.min(fixed ))/(np.max(fixed ) - np.min(fixed ))

if inshape[0]>128 or inshape[1]>128 or inshape[2]>128:
    if args.slidingwindow:
        # Method1: slice image to obtain 128x128x128 chunks. 
        window_size = np.array([128,128,128])
        overlap = np.array([16,16,16])
        window_shift = window_size -  overlap
        window_stride = 1
        window_num = np.ceil((inshape - overlap) / window_shift)
        padded_size = (window_num * window_shift + overlap).astype(np.int16)
        padded_fixed =  np.pad(fixed,  [[0,0],[0, padded_size[0]-inshape[0]], [0, padded_size[1]-inshape[1]],[0,padded_size[2]-inshape[2]],[0,0]])
        padded_moving = np.pad(moving, [[0,0],[0, padded_size[0]-inshape[0]], [0, padded_size[1]-inshape[1]],[0,padded_size[2]-inshape[2]],[0,0]])
        xx, yy, zz = np.mgrid[0:window_num[0], 0:window_num[1], 0:window_num[2]]
        grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T.astype(np.int16)
        grid_coordinates = [ [(window_shift[0]*x),  overlap[0]+window_shift[0]*(x+1), 
                              (window_shift[1]*y),  overlap[1]+window_shift[1]*(y+1),
                              (window_shift[2]*z),  overlap[2]+window_shift[2]*(z+1)] for x,y,z in grid]
        grid_count = len(grid_coordinates)
        fixed_grid_list=[padded_fixed[g[0]:g[1],g[2]:g[3],g[4]:g[5]] for g in grid_coordinates]
        moving_grid_list=[padded_moving[g[0]:g[1],g[2]:g[3],g[4]:g[5]] for g in grid_coordinates]
        padded_warp =np.zeros([padded_fixed.shape[0],padded_fixed.shape[1],padded_fixed.shape[2],padded_fixed.shape[3],3])
        config = dict(inshape=window_size, input_model=None)
        tf.config.run_functions_eagerly(True)
        with tf.device(device):
            print("loading synmorph model...")
            synthmorph = vxm.networks.VxmDense.load(args.model, **config)
            for i, g in enumerate(grid_coordinates):
                # load model and predict
                print("Registering grid#{}/{}".format(i+1, grid_count))
                warp_patch = synthmorph.register(padded_moving[:,g[0]:g[1],g[2]:g[3],g[4]:g[5],:], padded_fixed[:,g[0]:g[1],g[2]:g[3],g[4]:g[5],:])
                padded_warp[:,g[0]:g[1],g[2]:g[3],g[4]:g[5],:] = warp_patch
        warp = padded_warp[:,0:inshape[0],0:inshape[1],0:inshape[2],:]
        vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

        print("done registration, now perform transform")
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([orig_moving, warp])
    elif args.shrink:
        # Method2: shirnk images to fit 128x128x128, calculate warp, then re-expand warp to fit original size, then performe warping.
        dim_max = np.max(inshape)
        re_size = np.ceil((inshape/dim_max)*128).astype(np.int16)
        re_fixed  = skimage.transform.resize_local_mean(fixed,  [1, re_size[0], re_size[1], re_size[2],1])
        re_moving = skimage.transform.resize_local_mean(moving, [1, re_size[0], re_size[1], re_size[2],1])
        re_fixed = np.pad(re_fixed, [[0,0],[0,128-re_size[0]],[0,128-re_size[1]],[0,128-re_size[2]],[0,0]])
        re_moving = np.pad(re_moving, [[0,0],[0,128-re_size[0]],[0,128-re_size[1]],[0,128-re_size[2]],[0,0]])
        with tf.device(device):
            config = dict(inshape=[128,128,128], input_model=None)
            synthmorph = vxm.networks.VxmDense.load(args.model, **config)
            re_warp = synthmorph.register(re_moving, re_fixed)
            warp = skimage.transform.resize_local_mean(re_warp, [1, dim_max, dim_max, dim_max,3])*(dim_max/128)  # displacement also increased after upsizing.
            warp = warp[:,0:inshape[0],0:inshape[1],0:inshape[2],:]
            print("done registration, now perform transform")
            moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([orig_moving, warp])
else:
    with tf.device(device):
        # load model and predict
        config = dict(inshape=inshape, input_model=None)
        warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
        print("done registration, now perform transform")
        moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([orig_moving, warp])


print("done transform, save images.")
# save warp
if args.warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)
