#!/usr/bin/env python3
"""
================================================================================
DTI PREPROCESSING PIPELINE 2: THE DIPY BIOPHYSICAL MULTI-COMPARTMENT PARADIGM
================================================================================
Pipeline ID: dti_pip2.py
Philosophy : Leverages DIPY's advanced physical modeling, including Rician noise
             unbiasing and automated initial Otsu brain tissue separation.
Stages     :
  1. Format Ingestion & Gradient Table Construction (DIPY GradientTable)
  2. Multi-Slice Gibbs Ringing Removal (Kellner et al.)
  3. MP-PCA Denoising + Analytical Rician Noise Floor Floor-Correction
  4. Residual Difference Calculation & Signal Metric Audit (QC Gate 1)
  5. Initial Otsu Brain Mask Generation + b0 Extraction for Manual Refinement
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.denoise.gibbs import gibbs_removal
from dipy.denoise.localpca import mppca
from dipy.segment.mask import median_otsu

WORKSPACE_DIR = r"e:\DTI\FB34_MRI\FB34_3T_MRI_inSKULL\FB34_3T_MRI_inSKULL"
DEFAULT_DWI = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.nii.gz")
DEFAULT_BVAL = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.bval")
DEFAULT_BVEC = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.bvec")
ITK_SNAP_EXE = r"C:\Program Files\ITK-SNAP 4.4\bin\ITK-SNAP.exe"

def run_pipeline(dwi_path, bval_path, bvec_path, out_dir):
    print("=" * 80)
    print(" [PIPELINE 2] EX VIVO FETAL DTI PREPROCESSING: DIPY BIOPHYSICAL PARADIGM")
    print("=" * 80)
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # STAGE 1: INGESTION & GRADIENT TABLE SETUP
    # --------------------------------------------------------------------------
    print(f"\n[Stage 1/5] Ingesting Data & Constructing DIPY Gradient Table...")
    img = nib.load(dwi_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header

    gtab = gradient_table(bval_path, bvec_path)
    print(f"  -> Data Shape: {data.shape}")
    print(f"  -> Total Volumes: {len(gtab.bvals)} (b0: {np.sum(gtab.b0s_mask)}, DWI: {np.sum(~gtab.b0s_mask)})")
    print(f"  -> b-values: {np.unique(gtab.bvals.astype(int))}")

    # --------------------------------------------------------------------------
    # STAGE 2: GIBBS RINGING CORRECTION
    # --------------------------------------------------------------------------
    print(f"\n[Stage 2/5] Performing Kellner Gibbs Ringing Removal...")
    unringed_data = np.zeros_like(data)
    for v in range(data.shape[3]):
        unringed_data[..., v] = gibbs_removal(data[..., v], slice_axis=2)

    unringed_path = os.path.join(out_dir, "pip2_dwi_unringed.nii.gz")
    nib.save(nib.Nifti1Image(unringed_data, affine, header), unringed_path)
    print(f"  -> Saved: {unringed_path}")

    # --------------------------------------------------------------------------
    # STAGE 3: MP-PCA DENOISING + RICIAN NOISE-FLOOR CORRECTION
    # --------------------------------------------------------------------------
    print(f"\n[Stage 3/5] Computing MP-PCA Denoising & Removing Rician Noise Floor...")
    denoised_data, sigma = mppca(unringed_data, patch_radius=2, return_sigma=True)

    # Unbias Rician magnitude floor: S = sqrt(max(0, M^2 - 2*sigma^2))
    # Broaden sigma across the 4th dimension for vector broadcasting
    sigma_4d = sigma[..., np.newaxis]
    rician_clean_data = np.sqrt(np.maximum(0.0, denoised_data**2 - 2.0 * (sigma_4d**2)))

    clean_path = os.path.join(out_dir, "pip2_dwi_rician_clean.nii.gz")
    sigma_path = os.path.join(out_dir, "pip2_noise_sigma.nii.gz")
    nib.save(nib.Nifti1Image(rician_clean_data, affine, header), clean_path)
    nib.save(nib.Nifti1Image(sigma, affine), sigma_path)
    print(f"  -> Saved Cleaned DWI: {clean_path}")
    print(f"  -> Saved Noise Sigma: {sigma_path}")

    # --------------------------------------------------------------------------
    # STAGE 4: RESIDUAL MAP GENERATION (QC GATE 1)
    # --------------------------------------------------------------------------
    print(f"\n[Stage 4/5] Generating Denoising Residual Map...")
    residual = data - rician_clean_data
    residual_path = os.path.join(out_dir, "pip2_denoise_residual.nii.gz")
    nib.save(nib.Nifti1Image(residual, affine, header), residual_path)

    print(f"  -> Saved: {residual_path}")
    print(f"  -> Residual Mean: {np.mean(residual):.4f}, StdDev: {np.std(residual):.4f}")

    # --------------------------------------------------------------------------
    # STAGE 5: INITIAL OTSU MASKING + NATIVE b0 FOR MANUAL REFINEMENT
    # --------------------------------------------------------------------------
    print(f"\n[Stage 5/5] Extracting b0 & Generating Initial Median-Otsu Brain Mask...")
    b0_indices = np.where(gtab.b0s_mask)[0]
    mean_b0 = np.mean(rician_clean_data[..., b0_indices], axis=3) if len(b0_indices) > 0 else rician_clean_data[..., 0]

    b0_path = os.path.join(out_dir, "pip2_b0_native.nii.gz")
    nib.save(nib.Nifti1Image(mean_b0, affine), b0_path)

    # Run DIPY Median Otsu for automated baseline
    _, initial_mask = median_otsu(mean_b0, median_radius=2, numpass=1)
    otsu_mask_path = os.path.join(out_dir, "pip2_otsu_initial_mask.nii.gz")
    nib.save(nib.Nifti1Image(initial_mask.astype(np.uint8), affine), otsu_mask_path)
    print(f"  -> Saved Native b0: {b0_path}")
    print(f"  -> Saved Initial Otsu Mask: {otsu_mask_path}")

    print("\n" + "=" * 80)
    print(" [PIPELINE 2] PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNEXT MANDATORY MANUAL STEPS (ITK-SNAP):")
    print(f" 1. Inspect Residual (QC Gate 1):")
    print(f"    & \"{ITK_SNAP_EXE}\" \"{residual_path}\"")
    print(f" 2. Refine Brain Mask:")
    print(f"    & \"{ITK_SNAP_EXE}\" -g \"{b0_path}\" -s \"{otsu_mask_path}\"")
    print(f"    -> Use Eraser tool to remove extra-axial formalin fluid surrounding skull.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline 2: DIPY Biophysical Preprocessing")
    parser.add_argument("--dwi", default=DEFAULT_DWI, help="Path to raw 4D DWI NIfTI")
    parser.add_argument("--bval", default=DEFAULT_BVAL, help="Path to bval file")
    parser.add_argument("--bvec", default=DEFAULT_BVEC, help="Path to bvec file")
    parser.add_argument("--out_dir", default=os.path.join(WORKSPACE_DIR, "preproc_pip2_dipy_biophysical"), help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.dwi, args.bval, args.bvec, args.out_dir)
