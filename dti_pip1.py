#!/usr/bin/env python3
"""
================================================================================
DTI PREPROCESSING PIPELINE 1: THE FSL CLASSIC / STANDARD PARADIGM
================================================================================
Pipeline ID: dti_pip1.py
Philosophy : Classic, conservative neuroimaging workflow based on standard
             FSL/MRtrix conventions.
Stages     :
  1. Format Conversion & Ingestion (dcm2niix / NIfTI validation)
  2. Standard Gibbs Ringing Removal (Kellner algorithm, default isotropic)
  3. Standard MP-PCA Denoising (Isotropic 5x5x5 patch)
  4. Residual Difference Calculation & Quality Control (QC Gate 1)
  5. Native b0 Extraction & Preparation for Manual Brain Masking
================================================================================
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import nibabel as nib
from dipy.denoise.gibbs import gibbs_removal
from dipy.denoise.localpca import mppca

# Default paths tailored to FB34 workspace
WORKSPACE_DIR = r"e:\DTI\FB34_MRI\FB34_3T_MRI_inSKULL\FB34_3T_MRI_inSKULL"
DEFAULT_DWI = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.nii.gz")
DEFAULT_BVAL = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.bval")
DEFAULT_BVEC = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.bvec")
DCM2NIIX_EXE = r"C:\Users\Admin\neuro_tools\bin\dcm2niix.exe"
ITK_SNAP_EXE = r"C:\Program Files\ITK-SNAP 4.4\bin\ITK-SNAP.exe"

def run_pipeline(dwi_path, bval_path, bvec_path, out_dir):
    print("=" * 80)
    print(" [PIPELINE 1] EX VIVO FETAL DTI PREPROCESSING: FSL CLASSIC PARADIGM")
    print("=" * 80)
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # STAGE 1: INGESTION & DATA VALIDATION
    # --------------------------------------------------------------------------
    print(f"\n[Stage 1/5] Loading and Validating Input Data...")
    if not os.path.exists(dwi_path):
        raise FileNotFoundError(f"DWI file not found: {dwi_path}")
    
    img = nib.load(dwi_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header

    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    num_b0 = np.sum(bvals < 50)
    num_dwi = np.sum(bvals >= 50)
    print(f"  -> Dimensions: {data.shape}")
    print(f"  -> Voxel Spacing: {img.header.get_zooms()[:3]} mm")
    print(f"  -> Baseline b0 volumes: {num_b0}")
    print(f"  -> Diffusion directions: {num_dwi}")

    # Copy bvals and bvecs to output directory for FSL
    out_bvals = os.path.join(out_dir, "bvals")
    out_bvecs = os.path.join(out_dir, "bvecs")
    np.savetxt(out_bvals, bvals, fmt="%d" if np.all(bvals == bvals.astype(int)) else "%.1f", delimiter=" ")
    np.savetxt(out_bvecs, bvecs.T, fmt="%.6f", delimiter=" ")

    # --------------------------------------------------------------------------
    # STAGE 2: GIBBS RINGING REMOVAL (Kellner Algorithm)
    # --------------------------------------------------------------------------
    print(f"\n[Stage 2/5] Running Standard Gibbs Ringing Removal (Kellner Algorithm)...")
    # Standard multi-slice unringing
    unringed_data = np.zeros_like(data)
    for v in range(data.shape[3]):
        unringed_data[..., v] = gibbs_removal(data[..., v], slice_axis=2)
    
    unringed_path = os.path.join(out_dir, "pip1_dwi_unringed.nii.gz")
    nib.save(nib.Nifti1Image(unringed_data, affine, header), unringed_path)
    print(f"  -> Saved: {unringed_path}")

    # --------------------------------------------------------------------------
    # STAGE 3: STANDARD MP-PCA DENOISING (Isotropic 5x5x5 Patch)
    # --------------------------------------------------------------------------
    print(f"\n[Stage 3/5] Running Standard MP-PCA Denoising (patch_radius=2, 5x5x5)...")
    denoised_data, sigma = mppca(unringed_data, patch_radius=2, return_sigma=True)
    
    denoised_path = os.path.join(out_dir, "pip1_dwi_denoised.nii.gz")
    sigma_path = os.path.join(out_dir, "pip1_noise_sigma.nii.gz")
    nib.save(nib.Nifti1Image(denoised_data, affine, header), denoised_path)
    nib.save(nib.Nifti1Image(sigma, affine), sigma_path)
    print(f"  -> Saved: {denoised_path}")
    print(f"  -> Saved: {sigma_path}")

    # --------------------------------------------------------------------------
    # STAGE 4: RESIDUAL CALCULATION & QUALITY CONTROL (QC GATE 1)
    # --------------------------------------------------------------------------
    print(f"\n[Stage 4/5] Computing Denoising Residual Map...")
    residual = data - denoised_data
    residual_path = os.path.join(out_dir, "pip1_denoise_residual.nii.gz")
    nib.save(nib.Nifti1Image(residual, affine, header), residual_path)
    
    res_mean = np.mean(residual)
    res_std = np.std(residual)
    print(f"  -> Saved: {residual_path}")
    print(f"  -> Residual Mean: {res_mean:.4f}, StdDev: {res_std:.4f}")

    # --------------------------------------------------------------------------
    # STAGE 5: EXTRACT NATIVE b0 FOR MANUAL BRAIN MASKING
    # --------------------------------------------------------------------------
    print(f"\n[Stage 5/5] Extracting Mean b0 Volume for Manual Brain Masking...")
    b0_indices = np.where(bvals < 50)[0]
    if len(b0_indices) > 0:
        mean_b0 = np.mean(denoised_data[..., b0_indices], axis=3)
    else:
        mean_b0 = denoised_data[..., 0]

    b0_path = os.path.join(out_dir, "pip1_b0_native.nii.gz")
    nib.save(nib.Nifti1Image(mean_b0, affine), b0_path)
    print(f"  -> Saved: {b0_path}")

    print("\n" + "=" * 80)
    print(" [PIPELINE 1] PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNEXT MANDATORY MANUAL STEPS (ITK-SNAP):")
    print(f" 1. Inspect Residual (QC Gate 1):")
    print(f"    & \"{ITK_SNAP_EXE}\" \"{residual_path}\"")
    print(f"    -> Must show uniform, structureless static (zero brain outlines).")
    print(f" 2. Trace Brain Mask:")
    print(f"    & \"{ITK_SNAP_EXE}\" \"{b0_path}\"")
    print(f"    -> Trace outer Cortical Plate and save as 'native_brain_mask.nii.gz'.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline 1: FSL Classic Preprocessing")
    parser.add_argument("--dwi", default=DEFAULT_DWI, help="Path to raw 4D DWI NIfTI")
    parser.add_argument("--bval", default=DEFAULT_BVAL, help="Path to bval file")
    parser.add_argument("--bvec", default=DEFAULT_BVEC, help="Path to bvec file")
    parser.add_argument("--out_dir", default=os.path.join(WORKSPACE_DIR, "preproc_pip1_fsl_classic"), help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.dwi, args.bval, args.bvec, args.out_dir)
