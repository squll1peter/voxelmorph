import SimpleITK as sitk
import sys
import os 

def split_and_augment(vol, pair=None):
    '''pair is for paired CT volume, sharing same dimension and coordination'''
    pass

def split_and_augment_path(vol_pth, pair_pth=None):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(vol_pth)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = sitk.GetArrayFromImage(image)
    return img_array
