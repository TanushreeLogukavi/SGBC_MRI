# Ex Vivo Fetal Diffusion Tensor Imaging (DTI) Preprocessing Suite

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![DIPY](https://img.shields.io/badge/DIPY-1.12.1-brightgreen.svg)](https://dipy.org/)
[![NiBabel](https://img.shields.io/badge/NiBabel-5.4.2-orange.svg)](https://nipy.org/nibabel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A specialized computational suite providing **four distinct candidate preprocessing pipelines** for post-mortem human fetal brain diffusion MRI (dMRI/DTI). Specifically engineered to overcome the physical bottlenecks of ex vivo second-trimester fetal specimens scanned in-skull on clinical 3.0T MRI scanners (e.g., GE SIGNA Architect, 25 diffusion directions, $b=1000\ \text{s/mm}^2$).

---

## Table of Contents
- [1. Overview & Biophysical Background](#1-overview--biophysical-background)
- [2. Repository Architecture](#2-repository-architecture)
- [3. The Four Preprocessing Pipelines](#3-the-four-preprocessing-pipelines)
- [4. Software Requirements & Installation](#4-software-requirements--installation)
- [5. Quickstart & Usage Instructions](#5-quickstart--usage-instructions)
- [6. The Human-in-the-Loop Protocol (ITK-SNAP Steps)](#6-the-human-in-the-loop-protocol-itk-snap-steps)
- [7. Benchmark Metrics & Selection Guide](#7-benchmark-metrics--selection-guide)
- [8. Citation & Acknowledgments](#8-citation--acknowledgments)

---

## 1. Overview & Biophysical Background

Ex vivo fetal diffusion MRI on a clinical scanner is one of the most challenging acquisition scenarios in neuroimaging. Standard adult processing workflows fail due to three compounding physical factors:

1. **Formalin Fixation Physics:** Tissue cross-linking dehydrates extracellular space and causes a 3- to 4-fold reduction in baseline diffusivity ($\text{MD} \approx 0.18\text{--}0.35 \times 10^{-3}\ \text{mm}^2/\text{s}$). A nominal $b=1000\ \text{s/mm}^2$ acts like an in vivo $b \approx 250\text{--}300\ \text{s/mm}^2$, yielding low diffusion attenuation and low SNR.
2. **Unmyelinated Fetal Microstructure:** Second-trimester white matter pathways have zero myelin insulation. Radial Diffusivity (RD) is nearly equal to Axial Diffusivity (AD). Baseline Fractional Anisotropy (FA) is naturally low ($0.12\text{--}0.25$); applying standard adult thresholds ($\text{FA} > 0.20$) erases the tracts.
3. **Anisotropic Voxels & Immersion Fluid Contamination:** A 2.0 mm slice thickness causes through-plane partial volume averaging across thin developmental zones (Cortical Plate $<1.0\text{ mm}$ thick). Furthermore, high-diffusivity formalin fluid ($\text{MD} > 2.0 \times 10^{-3}\ \text{mm}^2/\text{s}$) surrounding the unossified fetal skull bleeds into edge voxels if not strictly excluded.

This repository provides four standalone, mathematically principled pipelines designed to benchmark and resolve these bottlenecks.

---

## 2. Repository Architecture

```text
├── dti_pip1.py                                       # Pipeline A: FSL Classic / Standard Neuroimaging Paradigm
├── dti_pip2.py                                       # Pipeline B: DIPY Biophysical Multi-Compartment Paradigm
├── dti_pip3.py                                       # Pipeline C: DSI Studio Model-Free Source Prep Paradigm
├── dti_pip4.py                                       # Pipeline D: Consortium High-Res Hybrid Paradigm (RECOMMENDED)
└── README.md                                         # Repository documentation
```

---

## 3. The Four Preprocessing Pipelines

| Pipeline File | Name & Paradigm | Key Preprocessing Innovations | Best Used For |
| :--- | :--- | :--- | :--- |
| **`dti_pip1.py`** | **Pipeline A: FSL Classic** | Standard Kellner Gibbs unringing + Isotropic $5\times5\times5$ MP-PCA denoising. Standardizes FSL format. | Establishing a conservative, traditional clinical neuroimaging baseline. |
| **`dti_pip2.py`** | **Pipeline B: DIPY Biophysical** | DIPY GradientTable integration + Kellner unringing + Rician noise-floor pedestal correction + Otsu initialization. | Biophysical modeling, Free-Water Elimination (FWE-DTI), and native Python workflows. |
| **`dti_pip3.py`** | **Pipeline C: DSI Studio Prep** | B-table generation + signal conditioning + prepares source inputs for continuous Spin Distribution Function (SDF). | Fast interactive 3D visual reconstruction and deterministic DTI tracking in DSI Studio. |
| **`dti_pip4.py`** | **Pipeline D: Consortium Hybrid** *(Recommended)* | **In-plane Gibbs unringing (axes 0,1)** + **Expanded-kernel MP-PCA ($7\times7\times5$)** to compensate for GE zero-fill recon + **Rician noise unbiasing** + institutional mask integration. | **Highest accuracy for GE 3T 25-direction in-skull fetal scans.** Maximizes Cortical Radiality ($RI \ge 0.85$). |

---

## 4. Software Requirements & Installation

### Core Dependencies (Native Windows or Linux)
* **Python 3.11+**
* **NiBabel** $\ge 5.4.0$ (NIfTI I/O)
* **DIPY** $\ge 1.12.0$ (Diffusion imaging in Python)
* **NumPy** $\ge 1.24.0$ & **SciPy** $\ge 1.10.0$

### Visualization & Manual Annotation Tools
* **ITK-SNAP 4.x** (Required for manual residual visual audit and hand-drawn brain masking): [Download ITK-SNAP](http://www.itksnap.org/)
* **dcm2niix** (For raw DICOM to 4D NIfTI conversion): [Download dcm2niix](https://github.com/rordenlab/dcm2niix)
* **DSI Studio** (Optional, for Pipeline C): [Download DSI Studio](http://dsi-studio.labsolver.org/)

### Quick Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/exvivo-fetal-dti-pipelines.git
cd exvivo-fetal-dti-pipelines

# 2. Install scientific dependencies
pip install numpy scipy matplotlib nibabel dipy pandas pydicom tqdm
```

---

## 5. Quickstart & Usage Instructions

All four scripts are fully standalone and can be executed via terminal/PowerShell. They accept optional CLI arguments (`--dwi`, `--bval`, `--bvec`, `--out_dir`) and automatically default to processing local repository data if no arguments are passed.

### Run Pipeline 1 (FSL Classic Paradigm)
```bash
python dti_pip1.py --dwi path/to/dwi.nii.gz --bval path/to/dwi.bval --bvec path/to/dwi.bvec --out_dir ./preproc_pip1
```
*Outputs:* `pip1_dwi_unringed.nii.gz`, `pip1_dwi_denoised.nii.gz`, `pip1_denoise_residual.nii.gz`, `pip1_b0_native.nii.gz`, and standardized FSL `bvals`/`bvecs`.

### Run Pipeline 2 (DIPY Biophysical Multi-Compartment Paradigm)
```bash
python dti_pip2.py --dwi path/to/dwi.nii.gz --bval path/to/dwi.bval --bvec path/to/dwi.bvec --out_dir ./preproc_pip2
```
*Outputs:* `pip2_dwi_rician_clean.nii.gz`, `pip2_noise_sigma.nii.gz`, `pip2_denoise_residual.nii.gz`, `pip2_b0_native.nii.gz`, and `pip2_otsu_initial_mask.nii.gz`.

### Run Pipeline 3 (DSI Studio Model-Free Paradigm)
```bash
python dti_pip3.py --dwi path/to/dwi.nii.gz --bval path/to/dwi.bval --bvec path/to/dwi.bvec --out_dir ./preproc_pip3
```
*Outputs:* `pip3_dwi_conditioned.nii.gz`, `b_table.txt` (4-column DSI Studio format), `pip3_conditioning_residual.nii.gz`, and `pip3_b0_native.nii.gz`.

### Run Pipeline 4 (Consortium Hybrid Gold Standard - Recommended)
```bash
python dti_pip4.py --dwi path/to/dwi.nii.gz --bval path/to/dwi.bval --bvec path/to/dwi.bvec --out_dir ./preproc_pip4
```
*Outputs:* `pip4_dwi_hybrid_clean.nii.gz`, `pip4_noise_sigma.nii.gz`, `pip4_residual_hybrid.nii.gz`, and `pip4_b0_native.nii.gz`.

---

## 6. The Human-in-the-Loop Protocol (ITK-SNAP Steps)

Ex vivo fetal neuroimaging cannot be 100% automated because unossified cranial cartilage and surrounding formalin fluid create severe partial volume artifacts. **Two manual quality-control checkpoints must be performed in ITK-SNAP after running any pipeline script:**

```text
[Pipeline Script (Automated)]
              │
              ▼
   QC Gate 1: Residual Audit (ITK-SNAP - 30 sec)
   ├── PASS: Uniform Gaussian/Rician "TV static"
   └── FAIL: Anatomical structures visible -> Adjust patch size
              │
              ▼
   Step 5: Hand-Drawn Brain Masking (ITK-SNAP - 15-20 min)
   ├── Trace outer boundary of the Cortical Plate
   └── Strictly exclude 100% of surrounding formalin immersion fluid
              │
              ▼
   [Ready for FSL eddy / Tensor Estimation / Tractography]
```

### Manual Action 1: Residual Map Inspection (QC Gate 1)
Open the generated residual image in ITK-SNAP:
```bash
itksnap ./preproc_pip4/pip4_residual_hybrid.nii.gz
```
* **PASS Criteria:** Uniform, structureless white Gaussian/Rician noise across all slices.
* **FAIL Criteria:** Fetal ventricles, eyes, or cortical mantle edges visible in the residual. If seen, signal was erroneously deleted.

### Manual Action 2: Hand-Drawn Brain Masking (`native_brain_mask.nii.gz`)
Open the extracted native $b=0$ volume in ITK-SNAP:
```bash
itksnap ./preproc_pip4/pip4_b0_native.nii.gz
```
1. Select the **Paintbrush Tool** (Radius: 2–3 voxels).
2. Slice-by-slice on the axial view, hand-trace the outer boundary of the brain parenchyma.
3. **The Golden Rule:** Hug the outer margin of the Cortical Plate. **Exclude 100% of the bright formalin immersion fluid** sitting between the brain and the skull.
4. Save the segmentation as `native_brain_mask.nii.gz` in the output folder.

---

## 7. Benchmark Metrics & Selection Guide

To determine which pipeline performs best on your dataset, evaluate results against the **Six Objective Metrics** defined in `DTI Pipeline Benchmarking & Decision Framework.docx`:

| Evaluation Metric | Formula / Criterion | Target for GA 21 Weeks |
| :--- | :--- | :--- |
| **1. Eigenvalue Validity (EVF)** | Percentage of voxels with $\lambda_1 \ge \lambda_2 \ge \lambda_3 > 0$ | $\ge 99.8\%$ (Must have zero negative eigenvalues) |
| **2. Cortical Radiality Index (RI)** | $\text{RI} = \|\vec{n}_{\text{pial}} \cdot \vec{V}_1\|$ | $\ge 0.80$ (Radial glial perpendicular scaffolding) |
| **3. CP-to-Subplate Contrast (CNR)** | $\frac{\|\text{FA}_{\text{CP}} - \text{FA}_{\text{SP}}\|}{\sqrt{\sigma_{\text{CP}}^2 + \sigma_{\text{SP}}^2}}$ | Higher = sharper laminar boundary definition |
| **4. Residual Fitting Error (RMSE)** | Root-mean-square discrepancy across 25 directions | Lower = superior fit to raw physical measurements |
| **5. CST False-Positive Rate** | Non-anatomical streamlines entering ventricles / midline | $< 2.0\%$ |
| **6. Target Registration Error (TRE)** | Distance from 0.6 mm 3D T2 anatomical landmarks | $\le 0.6\text{ mm}$ (Sub-voxel alignment via ANTs PPD) |

### Head-to-Head Tournament Summary (FB34 Reference):
* **Pipeline 4 (Consortium Hybrid)** wins decisively across all biological criteria: it achieves the highest Cortical Radiality ($\text{RI} = 0.85 \pm 0.05$), sharpest layer separation ($\text{CNR} = 3.42$), lowest residual error ($\text{RMSE} = 0.039$), and lowest false-positive tract rate ($1.5\%$).

---

## 8. Citation & Acknowledgments

If you use these pipelines or the associated SOP protocols in your research, please cite:

1. **MP-PCA Denoising:** Veraart J, et al. *Denoising of diffusion MRI using random matrix theory.* NeuroImage, 2016.
2. **Gibbs Ringing Removal:** Kellner E, et al. *Gibbs-ringing artifact removal based on local subvoxel-shifts.* Magn Reson Med, 2016.
3. **FSL Eddy & Outlier Replacement:** Andersson JLR, Sotiropoulos SN. *An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging.* NeuroImage, 2016.
4. **DIPY:** Garyfallidis E, et al. *DIPY, a library for the analysis of diffusion MRI data.* Frontiers in Neuroinformatics, 2014.
5. **Fetal Brain Laminar Histology:** Kostović I, Judaš M. *The development of the subplate and cortical plate in the human fetal cerebrum.* Acta Paediatr, 2010.

---

### License
This project is licensed under the MIT License - see the `LICENSE` file for details.

### Created by Tanushree L for enquiries contact tanushreelogukavi@gmail.com
