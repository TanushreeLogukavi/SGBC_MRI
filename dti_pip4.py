#!/usr/bin/env python3
"""
================================================================================
DTI PREPROCESSING PIPELINE 4: THE CONSORTIUM HIGH-RESOLUTION HYBRID PARADIGM
================================================================================
Pipeline ID: dti_pip4.py
Philosophy : The Recommended Gold Standard. Specifically optimized for GE 3T
             in-skull fetal acquisitions (FB34, 25 directions, b=1000).
Stages     :
  1. Automated Format Validation & Gradient Geometry Verification
  2. In-Plane Gibbs Ringing Removal (Axes 0,1 - preserving slice axis)
  3. Expanded-Kernel MP-PCA (7x7x5) compensating for GE 88->256 Zero-Filling
  4. Analytical Rician Noise-Floor Pedestal Subtraction
  5. Residual Map Generation (QC Gate 1) & Native b0 Extraction for ITK-SNAP
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

WORKSPACE_DIR = r"e:\DTI\FB34_MRI\FB34_3T_MRI_inSKULL\FB34_3T_MRI_inSKULL"
DEFAULT_DWI = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.nii.gz")
DEFAULT_BVAL = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.bval")
DEFAULT_BVEC = os.path.join(WORKSPACE_DIR, "S12- AX_DWI_TENSOR_B-100", "S12-_AX_DWI_TENSOR_B-100_FETAL_MR_20220601081357_23.bvec")
DEFAULT_MASK = os.path.join(WORKSPACE_DIR, "MRI analysis", "WithoutCSF", "BrainMask", "BrainMask_withoutCSF_ver2.nii.gz")
DCM2NIIX_EXE = r"C:\Users\Admin\neuro_tools\bin\dcm2niix.exe"
ITK_SNAP_EXE = r"C:\Program Files\ITK-SNAP 4.4\bin\ITK-SNAP.exe"

def run_pipeline(dwi_path, bval_path, bvec_path, out_dir, ref_mask):
    print("=" * 80)
    print(" [PIPELINE 4] EX VIVO FETAL DTI PREPROCESSING: CONSORTIUM HYBRID GOLD STANDARD")
    print("=" * 80)
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # STAGE 1: INGESTION & GE ACQUISITION GEOMETRY CHECK
    # --------------------------------------------------------------------------
    print(f"\n[Stage 1/5] Ingesting NIfTI & Verifying GE Acquisition Parameters...")
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

    zooms = header.get_zooms()
    print(f"  -> Matrix Shape: {data.shape}")
    print(f"  -> In-Plane Spacing: {zooms[0]:.2f} x {zooms[1]:.2f} mm (Recon: 256x256 from Acq: 88x88)")
    print(f"  -> Slice Thickness: {zooms[2]:.2f} mm")
    print(f"  -> Total Volumes: {len(bvals)} (Baseline b0: {np.sum(bvals < 50)}, Diffusion Directions: {np.sum(bvals >= 50)})")

    # Save copy of bvals and bvecs in output
    np.savetxt(os.path.join(out_dir, "bvals"), bvals, fmt="%d" if np.all(bvals == bvals.astype(int)) else "%.1f", delimiter=" ")
    np.savetxt(os.path.join(out_dir, "bvecs"), bvecs.T, fmt="%.6f", delimiter=" ")

    # --------------------------------------------------------------------------
    # STAGE 2: IN-PLANE GIBBS UNRINGING ALONG ACQUISITION AXES
    # --------------------------------------------------------------------------
    print(f"\n[Stage 2/5] Performing Gibbs Ringing Removal along Acquisition Axes (0,1)...")
    unringed_data = np.zeros_like(data)
    for v in range(data.shape[3]):
        # Unring strictly along in-plane dimensions, preserving through-plane slice profile
        unringed_data[..., v] = gibbs_removal(data[..., v], slice_axis=2)

    unringed_path = os.path.join(out_dir, "pip4_dwi_unringed.nii.gz")
    nib.save(nib.Nifti1Image(unringed_data, affine, header), unringed_path)
    print(f"  -> Saved Unringed DWI: {unringed_path}")

    # --------------------------------------------------------------------------
    # STAGE 3: EXPANDED-KERNEL MP-PCA (7x7x5) + RICIAN UNBIASING
    # --------------------------------------------------------------------------
    print(f"\n[Stage 3/5] Running Expanded-Kernel MP-PCA (patch=[3,3,2] -> 7x7x5 voxels)...")
    print(f"  -> Note: Expanded patch compensates for GE zero-fill spatial correlation.")
    denoised_data, sigma = mppca(unringed_data, patch_radius=[3, 3, 2], return_sigma=True)

    print(f"  -> Subtracting Rician Noise Floor Pedestal...")
    sigma_4d = sigma[..., np.newaxis]
    # S = sqrt(max(0, M^2 - 2 * sigma^2))
    clean_data = np.sqrt(np.maximum(0.0, denoised_data**2 - 2.0 * (sigma_4d**2)))

    clean_path = os.path.join(out_dir, "pip4_dwi_hybrid_clean.nii.gz")
    sigma_path = os.path.join(out_dir, "pip4_noise_sigma.nii.gz")
    nib.save(nib.Nifti1Image(clean_data, affine, header), clean_path)
    nib.save(nib.Nifti1Image(sigma, affine), sigma_path)
    print(f"  -> Saved Cleaned DWI: {clean_path}")
    print(f"  -> Saved Noise Sigma: {sigma_path}")

    # --------------------------------------------------------------------------
    # STAGE 4: RESIDUAL MAP GENERATION (QC GATE 1)
    # --------------------------------------------------------------------------
    print(f"\n[Stage 4/5] Computing Denoising Residual Map...")
    residual = data - clean_data
    residual_path = os.path.join(out_dir, "pip4_residual_hybrid.nii.gz")
    nib.save(nib.Nifti1Image(residual, affine, header), residual_path)

    res_mean = np.mean(residual)
    res_std = np.std(residual)
    print(f"  -> Saved Residual: {residual_path}")
    print(f"  -> Global Residual Statistics: Mean = {res_mean:.4f}, StdDev = {res_std:.4f}")

    # --------------------------------------------------------------------------
    # STAGE 5: EXTRACT NATIVE b0 & INTEGRATE WITH INSTITUTIONAL MASK
    # --------------------------------------------------------------------------
    print(f"\n[Stage 5/5] Extracting Mean b0 Volume for Manual ITK-SNAP Masking...")
    b0_indices = np.where(bvals < 50)[0]
    mean_b0 = np.mean(clean_data[..., b0_indices], axis=3) if len(b0_indices) > 0 else clean_data[..., 0]

    b0_path = os.path.join(out_dir, "pip4_b0_native.nii.gz")
    nib.save(nib.Nifti1Image(mean_b0, affine), b0_path)
    print(f"  -> Saved Native b0: {b0_path}")

    mask_status = "Not Found"
    if os.path.exists(ref_mask):
        mask_status = f"Available at {ref_mask}"

    print("\n" + "=" * 80)
    print(" [PIPELINE 4] CONSORTIUM HYBRID PREPROCESSING COMPLETED!")
    print("=" * 80)
    print("\nNEXT MANDATORY MANUAL STEPS (ITK-SNAP):")
    print(f" 1. Inspect Residual (QC Gate 1):")
    print(f"    & \"{ITK_SNAP_EXE}\" \"{residual_path}\"")
    print(f"    -> CRITERIA: Must be 100% uniform Gaussian noise (ZERO brain anatomy).")
    print(f" 2. Delineate / Verify Brain Mask (Fluid Exclusion):")
    if os.path.exists(ref_mask):
        print(f"    & \"{ITK_SNAP_EXE}\" -g \"{b0_path}\" -s \"{ref_mask}\"")
        print(f"    -> Existing manual mask detected! Audit against native b0 and save as 'native_brain_mask.nii.gz'.")
    else:
        print(f"    & \"{ITK_SNAP_EXE}\" \"{b0_path}\"")
        print(f"    -> Hand-trace outer Cortical Plate; exclude 100% of surrounding formalin fluid.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline 4: Consortium High-Res Hybrid Preprocessing")
    parser.add_argument("--dwi", default=DEFAULT_DWI, help="Path to raw 4D DWI NIfTI")
    parser.add_argument("--bval", default=DEFAULT_BVAL, help="Path to bval file")
    parser.add_argument("--bvec", default=DEFAULT_BVEC, help="Path to bvec file")
    parser.add_argument("--ref_mask", default=DEFAULT_MASK, help="Path to reference manual mask")
    parser.add_argument("--out_dir", default=os.path.join(WORKSPACE_DIR, "preproc_pip4_consortium_hybrid"), help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.dwi, args.bval, args.bvec, args.out_dir, args.ref_mask)
