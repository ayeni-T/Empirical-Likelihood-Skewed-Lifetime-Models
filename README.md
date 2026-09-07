# Empirical Likelihood Inference for Skewed Lifetime Models

## A Jackknife Approach for the QTEG Distribution

**Authors:** Taiwo Michael Ayeni and Yichuan Zhao
**Affiliation:** Department of Mathematics and Statistics, Georgia State University
**Year:** 2026

## Paper

| Field | Details |
|---|---|
| Title | Empirical Likelihood Inference for Skewed Lifetime Models: A Jackknife Approach for the QTEG Distribution |
| Authors | Taiwo Michael Ayeni and Yichuan Zhao |
| Status | Manuscript in preparation |
| Journal | To be announced upon acceptance |
| DOI | To be added upon publication |
| Preprint | To be added |

*This repository will be updated with the journal link and DOI upon acceptance and publication.*

## Overview

This repository contains the simulation and real-data analysis code for a paper
developing Jackknife Empirical Likelihood (JEL), Balanced Adjusted JEL (BAJEL),
and a new Variance-Calibrated JEL (VCJEL) for inference on the parameters and
survival function of the QTEG distribution. BAJEL corrects the small-sample
under-coverage of standard JEL for the shape and rate parameters; VCJEL
addresses the milder over-coverage of JEL for the survival function, a setting
in which BAJEL is not effective. The simulation study was conducted on the
Georgia State University ARCTIC HPC cluster.

## Repository Structure

```
├── QTEG_JEL_Arctic_v6.py          # Main simulation and real-data analysis script
├── QTEG_JEL_plot_results_v6.py    # Figure generation script — paper + supplementary
├── qteg_jel_array_v6.sh           # SLURM job array script for HPC execution
└── README.md                      # This file
```

## Requirements

- Python 3.8+
- numpy
- scipy
- pandas
- matplotlib

```
pip install numpy scipy pandas matplotlib
```

## Usage

### 1. Run Simulation (HPC / SLURM)

Submit the 36-block SLURM array job:

```
sbatch qteg_jel_array_v6.sh
```

### 2. Merge Results

After all 36 blocks complete:

```
python QTEG_JEL_Arctic_v6.py --merge
```

### 3. Real Data Analysis

```
python QTEG_JEL_Arctic_v6.py --realdata
```

### 4. Generate Figures

```
python QTEG_JEL_plot_results_v6.py
```

### 5. Test Run (20 replications, block 0)

```
python QTEG_JEL_Arctic_v6.py --test
```

## Simulation Design

| Feature | Details |
|---|---|
| Scenarios | 3 hazard regimes (decreasing, approximately constant, unimodal) |
| Sample sizes | n = 30, 50, 100, 200 |
| Nominal levels | 90%, 95%, 99% |
| Replications | N = 5,000 per configuration |
| Total blocks | 36 (parallelised via SLURM array) |
| Methods (α, β) | NA, JEL, BAJEL |
| Methods (S(t₀)) | NA, JEL, VCJEL |

## Real Datasets

| Dataset | n | t₀ | Source |
|---|---|---|---|
| Bladder cancer remission times | 128 | 4.0 months | Lee & Wang (2003) |
| Boeing 720 air conditioning failures | 213 | 50.0 hours | Proschan (1963) |
| Malignant melanoma survival | 205 | 1.5 years | Alizadeh et al. (2017) |
| Guinea pig survival times | 72 | 1.5 years | Bjerkedal (1960) |

All datasets are publicly available from their respective published sources.

## Methods Summary

| Method | Parameters | Survival S(t₀) |
|---|---|---|
| NA (Normal Approximation) | ✓ | ✓ (logit-transformed delta method) |
| JEL (Jackknife EL) | ✓ | ✓ |
| BAJEL (Balanced Adjusted JEL) | ✓ | — (persistent over-coverage at small n; not effective for this bounded functional) |
| VCJEL (Variance-Calibrated JEL) | — | ✓ (addresses JEL's over-coverage via variance-matching calibration) |

## Citation

If you use this code, please cite:

```bibtex
@article{AyeniZhao2026,
  author  = {Ayeni, Taiwo Michael and Zhao, Yichuan},
  title   = {Empirical Likelihood Inference for Skewed Lifetime Models:
             A Jackknife Approach for the {QTEG} Distribution},
  journal = {TBD},
  note    = {Manuscript in preparation},
  year    = {2026}
}
```

## Contact

Taiwo Michael Ayeni — tayeni2@gsu.edu
Yichuan Zhao (Corresponding Author) — yichuan@gsu.edu
Department of Mathematics and Statistics, Georgia State University, Atlanta, GA

## License

This code is made available for academic and research purposes.
