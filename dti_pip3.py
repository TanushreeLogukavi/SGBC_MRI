#!/usr/bin/env python3
"""
================================================================================
DTI PREPROCESSING PIPELINE 3: THE DSI STUDIO MODEL-FREE PARADIGM
================================================================================
Pipeline ID: dti_pip3.py
Philosophy : Prepares conditioned NIfTI and b-table datasets for interactive or
             batch DSI Studio reconstruction (GQI / DTI-RK4).
Stages     :
  1. Data Ingestion & B-Table Standardization
  2. Directional Outlier Identification & Gibbs Filtering
  3. Signal Conditioning & DSI Studio Source Preparation
  4. Residual Difference Calculation (QC Gate 1)
  5. Native b0 Extraction for DSI Studio / ITK-SNAP Mask Definition
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
DSI_STUDIO_EXE = r"C:\Users\Admin\neuro_tools\dsi_studio\dsi_studio_win\dsi_studio.exe"
ITK_SNAP_EXE = r"C:\Program Files\ITK-SNAP 4.4\bin\ITK-SNAP.exe"

def run_pipeline(dwi_path, bval_path, bvec_path, out_dir):
    print("=" * 80)
    print(" [PIPELINE 3] EX VIVO FETAL DTI PREPROCESSING: DSI STUDIO PARADIGM")
    print("=" * 80)
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # STAGE 1: INGESTION & B-TABLE GENERATION
    # --------------------------------------------------------------------------
    print(f"\n[Stage 1/5] Ingesting NIfTI and Generating DSI Studio B-Table...")
    img = nib.load(dwi_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header

    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T

    # Generate standard DSI Studio b_table.txt (4 columns: bval, bx, by, bz)
    btable_path = os.path.join(out_dir, "b_table.txt")
    btable = np.column_stack((bvals, bvecs))
    np.savetxt(btable_path, btable, fmt="%.4f", delimiter="\t")
    print(f"  -> Saved B-Table: {btable_path} ({len(bvals)} volumes)")

    # --------------------------------------------------------------------------
    # STAGE 2: GIBBS UNRINGING
    # --------------------------------------------------------------------------
    print(f"\n[Stage 2/5] Running Gibbs Ringing Removal...")
    unringed_data = np.zeros_like(data)
    for v in range(data.shape[3]):
        unringed_data[..., v] = gibbs_removal(data[..., v], slice_axis=2)

    unringed_path = os.path.join(out_dir, "pip3_dwi_unringed.nii.gz")
    nib.save(nib.Nifti1Image(unringed_data, affine, header), unringed_path)
    print(f"  -> Saved: {unringed_path}")

    # --------------------------------------------------------------------------
    # STAGE 3: PCA SIGNAL CONDITIONING
    # --------------------------------------------------------------------------
    print(f"\n[Stage 3/5] Applying Signal Conditioning & Noise Suppression...")
    conditioned_data, sigma = mppca(unringed_data, patch_radius=2, return_sigma=True)
    
    conditioned_path = os.path.join(out_dir, "pip3_dwi_conditioned.nii.gz")
    nib.save(nib.Nifti1Image(conditioned_data, affine, header), conditioned_path)
    print(f"  -> Saved Conditioned DWI: {conditioned_path}")

    # --------------------------------------------------------------------------
    # STAGE 4: RESIDUAL MAP CALCULATION (QC GATE 1)
    # --------------------------------------------------------------------------
    print(f"\n[Stage 4/5] Computing Conditioning Residual Map...")
    residual = data - conditioned_data
    residual_path = os.path.join(out_dir, "pip3_conditioning_residual.nii.gz")
    nib.save(nib.Nifti1Image(residual, affine, header), residual_path)
    print(f"  -> Saved: {residual_path}")
    print(f"  -> Residual Mean: {np.mean(residual):.4f}, StdDev: {np.std(residual):.4f}")

    # --------------------------------------------------------------------------
    # STAGE 5: EXTRACT b0 & PREPARE DSI STUDIO WORKFLOW
    # --------------------------------------------------------------------------
    print(f"\n[Stage 5/5] Extracting Native b0 & Preparing DSI Studio Command...")
    b0_indices = np.where(bvals < 50)[0]
    mean_b0 = np.mean(conditioned_data[..., b0_indices], axis=3) if len(b0_indices) > 0 else conditioned_data[..., 0]

    b0_path = os.path.join(out_dir, "pip3_b0_native.nii.gz")
    nib.save(nib.Nifti1Image(mean_b0, affine), b0_path)
    print(f"  -> Saved Native b0: {b0_path}")

    # Save copy of bvals and bvecs in output
    np.savetxt(os.path.join(out_dir, "dwi.bval"), bvals, fmt="%d" if np.all(bvals == bvals.astype(int)) else "%.1f", delimiter=" ")
    np.savetxt(os.path.join(out_dir, "dwi.bvec"), bvecs.T, fmt="%.6f", delimiter=" ")

    print("\n" + "=" * 80)
    print(" [PIPELINE 3] PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNEXT MANDATORY MANUAL STEPS:")
    print(f" 1. Inspect Residual (QC Gate 1):")
    print(f"    & \"{ITK_SNAP_EXE}\" \"{residual_path}\"")
    print(f" 2. Launch DSI Studio for Masking & SRC Creation:")
    print(f"    & \"{DSI_STUDIO_EXE}\"")
    print(f"    -> Click 'Step 1: Open Source Images' -> Select '{conditioned_path}'.")
    print(f"    -> Load B-Table: '{btable_path}'.")
    print(f"    -> Under Masking, adjust threshold and use 3D brush to remove fluid.")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline 3: DSI Studio Preprocessing")
    parser.add_argument("--dwi", default=DEFAULT_DWI, help="Path to raw 4D DWI NIfTI")
    parser.add_argument("--bval", default=DEFAULT_BVAL, help="Path to bval file")
    parser.add_argument("--bvec", default=DEFAULT_BVEC, help="Path to bvec file")
    parser.add_argument("--out_dir", default=os.path.join(WORKSPACE_DIR, "preproc_pip3_dsi_studio"), help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.dwi, args.bval, args.bvec, args.out_dir)
