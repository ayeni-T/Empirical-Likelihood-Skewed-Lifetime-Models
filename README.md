# Empirical Likelihood Inference for Skewed Lifetime Models
## A Jackknife Approach for the QTEG Distribution

**Authors:** Taiwo Michael Ayeni and Yichuan Zhao  
**Affiliation:** Department of Mathematics and Statistics, Georgia State University  
**Year:** 2026

---

## Paper

| Field | Details |
|-------|----------|
| **Title** | Empirical Likelihood Inference for Skewed Lifetime Models: A Jackknife Approach for the QTEG Distribution |
| **Authors** | Taiwo Michael Ayeni and Yichuan Zhao |
| **Status** | Manuscript in preparation |
| **Journal** | To be announced upon acceptance |
| **DOI** | To be added upon publication |
| **Preprint** | To be added |

> This repository will be updated with the journal link and DOI upon acceptance and publication.

---

## Abstract

Wald-type confidence intervals may exhibit finite-sample miscalibration when the
underlying estimator distribution is skewed, a problem that commonly arises in
small samples from transformation-based lifetime models. This paper develops a
jackknife empirical likelihood (JEL) framework for the Quantile Transformed
Exponential-Gamma (QTEG) distribution, a flexible two-parameter lifetime model
whose shape parameter estimator exhibits noticeable right-skewness in small
samples. The leave-one-out structure of the QTEG log-likelihood reduces jackknife
pseudo-value computation to a one-dimensional root search, making JEL
computationally tractable. Three interval procedures are derived and studied:
standard JEL, adjusted JEL (AJEL) using two symmetric artificial observations to
expand the convex hull, and extended JEL (EJEL). Asymptotic chi-squared
calibration is established for all three under standard regularity conditions.
The framework is further extended to survival probability inference via a
logit-transformed Wald benchmark and empirical likelihood intervals for the
survival function S(t₀), and to joint bivariate confidence regions for the
parameter vector. A simulation study with N = 5,000 replications across three
hazard scenarios, four sample sizes, and three nominal levels show that AJEL
achieves near-nominal coverage across sample sizes and hazard regimes, while
Wald intervals over-cover and EJEL under-covers at small n. The methods are
illustrated on four real lifetime datasets, bladder cancer remission times,
Boeing 720 aircraft failure times, malignant melanoma survival, and guinea pig
survival  with the most pronounced differences arising in the heavily skewed
guinea pig dataset (n = 72, skewness = 2.54), where AJEL intervals for the shape
parameters are approximately 67% wider than the Wald interval, reflecting genuine
inferential uncertainty underrepresented by the symmetric normal approximation.

---

## Paper Contributions

1. **Computational tractability** — Jackknife pseudo-value computation for the
   QTEG model reduces to a one-dimensional root search, making JEL feasible
   without matrix inversion or high-dimensional optimisation.

2. **Three interval methods** — JEL, AJEL, and EJEL confidence intervals are
   derived for both the shape parameter α and the rate parameter β of the QTEG
   distribution, with full asymptotic theory established.

3. **Survival function inference** — The JEL framework is extended to inference
   on the survival probability S(t₀), including a logit-transformed Wald
   benchmark with full 2×2 covariance structure.

4. **Bivariate confidence regions** — Joint confidence regions for (α, β) are
   constructed via the bivariate JEL statistic.

5. **Finite-sample calibration** — Simulation evidence shows AJEL provides the
   best coverage calibration under skewness, correctly framing Wald over-coverage
   as miscalibration rather than superior performance.

---

## Statistical Background

### The QTEG Distribution

The Quantile Transformed Exponential-Gamma (QTEG) distribution arises from the
transformation Y = X² where X follows a Gamma(α, β) distribution. It is a
flexible two-parameter lifetime model that accommodates decreasing, approximately
constant, and increasing hazard functions depending on the shape parameter α:

- **α < 1:** decreasing hazard (infant-mortality regime)
- **α ≈ 1:** approximately constant hazard (random failure)
- **α > 1:** increasing hazard (wear-out regime)

The shape parameter estimator exhibits right-skewness in small samples, making
standard Wald intervals unreliable — the core motivation for the EL approach.

### Empirical Likelihood Methods

Empirical likelihood (Owen, 1988, 1990, 2001) is a nonparametric method for
constructing confidence intervals that automatically adapts to the shape of the
data distribution without assuming normality. The jackknife variant (Jing et al.,
2009) applies Owen's empirical likelihood to jackknife pseudo-values, enabling
computationally efficient inference for complex statistics.

| Method | Description | Key Property |
|--------|-------------|--------------|
| NA     | Normal approximation (Wald-type) | Symmetric by construction |
| JEL    | Jackknife Empirical Likelihood | Asymmetric, data-adaptive |
| AJEL   | Adjusted JEL (Emerson & Owen, 2009) | Expanded convex hull, best calibration |
| EJEL   | Extended JEL (Tsao & Wu, 2013) | Shortest intervals, slight under-coverage |

---

## Simulation Design

| Feature | Details |
|---------|---------|
| Scenarios | 3 hazard regimes: (α=1.5, β=0.5), (α=2.0, β=1.0), (α=3.0, β=2.0) |
| Sample sizes | n = 30, 50, 100, 200 |
| Nominal levels | 90%, 95%, 99% |
| Replications | N = 5,000 per configuration |
| Total configurations | 36 (parallelised via SLURM array) |
| HPC system | GSU ARCTIC cluster |
| Checkpointing | Every 50 replications (resume-safe) |

### Key Simulation Findings

- **AJEL** achieves near-nominal coverage across all scenarios and sample sizes
- **Wald (NA)** over-covers at small n due to positive finite-sample skewness in the estimator — this is miscalibration, not superiority
- **EJEL** produces the shortest intervals but under-covers at small n due to the convex-hull constraint
- **JEL** under-covers slightly at small n for the same reason
- All methods converge to the nominal level as n → ∞
- Survival function inference is more stable due to the bounded nature of S(t₀) ∈ (0,1)

---

## Real Data Application

| Dataset | n | t₀ | Source |
|---------|---|-----|--------|
| Bladder cancer remission times (months) | 128 | 4 months | Lee & Wang (2003) |
| Boeing 720 air conditioning failures (hours) | 213 | 50 hours | Proschan (1963) |
| Malignant melanoma survival (days) | 205 | 1.5 years | Alizadeh et al. (2017) |
| Guinea pig survival times (years) | 72 | 1.5 years | Bjerkedal (1960) |

The guinea pig dataset (DS4) is the most challenging case: skewness = 2.54,
CV = 0.75. AJEL intervals for α are approximately 67% wider than the Wald
interval and approximately 65% wider for β, reflecting genuine inferential
uncertainty underrepresented by the symmetric normal approximation.

All datasets are publicly available from their respective published sources.

---

## Repository Structure

```
├── QTEG_JEL_Arctic.py          # Main simulation and real-data analysis script
├── QTEG_JEL_plot_results.py    # Figure generation script (paper + supplementary)
├── qteg_jel_array.sh           # SLURM job array script for HPC execution
└── README.md                   # This file
```

---

## Requirements

- Python 3.8+
- numpy
- scipy
- pandas
- matplotlib

Install dependencies:
```bash
pip install numpy scipy pandas matplotlib
```

Or with conda:
```bash
conda install numpy scipy pandas matplotlib
```

---

## Usage

### 1. Run Simulation (HPC / SLURM)

The simulation is designed for a SLURM-based HPC cluster using a 36-block job
array (3 scenarios × 4 sample sizes × 3 nominal levels).

Before submitting, edit `qteg_jel_array.sh` and replace:
- `YOUR_ACCOUNT` with your HPC account name
- `YOUR_USERNAME` with your HPC username
- `your.email@institution.edu` with your email address
- `qteg_env` with your conda environment name

Submit the job array:
```bash
sbatch qteg_jel_array.sh
```

### 2. Merge Results

After all 36 blocks complete:
```bash
python QTEG_JEL_Arctic.py --merge
```

### 3. Real Data Analysis

```bash
python QTEG_JEL_Arctic.py --realdata
```

### 4. Generate Figures

```bash
python QTEG_JEL_plot_results.py
```

Figures are saved to the `results/` directory as both PDF (journal submission)
and PNG (300 dpi).

### 5. Test Run (single block, small simulation)

To verify the setup before submitting a full job array:
```bash
python QTEG_JEL_Arctic.py --test
```

---

## Output Files

After running the full pipeline, the `results/` directory contains:

**Simulation results:**
- `QTEG_JEL_block_*.json` — per-block checkpoint files
- `QTEG_JEL_full_results.csv` — merged simulation results

**Real data results:**
- `QTEG_JEL_realdata_results.json` — confidence intervals for all datasets

**Paper figures:**
- `fig_jel_sim_alpha.pdf/.png` — Coverage probability for α (Figure 1)
- `fig_jel_sim_beta.pdf/.png` — Coverage probability for β (Figure 2)
- `fig_jel_sim_surv.pdf/.png` — Coverage probability for S(t₀) (Figure 3)
- `fig_jel_realdata.pdf/.png` — Real data confidence intervals (Figure 4)

**Supplementary figures:**
- `fig_jel_alpha_al.pdf/.png` — Average interval length for α (Figure S1)
- `fig_jel_beta_al.pdf/.png` — Average interval length for β (Figure S2)
- `fig_jel_survival_al.pdf/.png` — Average interval length for S(t₀) (Figure S3)
- `fig_jel_alpha_cp_all_nominals.pdf/.png` — CP at all nominal levels for α (Figure S4)
- `fig_jel_survival_cp_all_nominals.pdf/.png` — CP at all nominal levels for S(t₀) (Figure S5)

---

## Key References

- Owen, A. B. (1988). Empirical likelihood ratio confidence intervals for a single functional. *Biometrika*, 75(2), 237–249. https://doi.org/10.1093/biomet/75.2.237
- Owen, A. B. (1990). Empirical likelihood ratio confidence regions. *The Annals of Statistics*, 18(1), 90–120. https://doi.org/10.1214/aos/1176347494
- Owen, A. B. (2001). *Empirical Likelihood*. Chapman & Hall/CRC, New York. https://doi.org/10.1201/9781420036152
- Jing, B.-Y., Yuan, J., and Zhou, W. (2009). Jackknife empirical likelihood. *Journal of the American Statistical Association*, 104(487), 1224–1232. https://doi.org/10.1198/jasa.2009.tm08260
- Emerson, S. C. and Owen, A. B. (2009). Calibration of the empirical likelihood method for a vector mean. *Electronic Journal of Statistics*, 3, 1161–1192. https://doi.org/10.1214/09-EJS518
- Chen, J., Variyath, A. M., and Abraham, B. (2008). Adjusted empirical likelihood and its properties. *Journal of Computational and Graphical Statistics*, 17(2), 426–443. https://doi.org/10.1198/106186008X321068
- Tsao, M. (2013). Extending the empirical likelihood by domain expansion. *The Canadian Journal of Statistics*, 41(2), 257–274. https://doi.org/10.1002/cjs.11175
- Tsao, M. and Wu, F. (2013). Empirical likelihood on the full parameter space. *The Annals of Statistics*, 41(4), 2176–2196. https://doi.org/10.1214/13-AOS1143
- Quenouille, M. H. (1956). Notes on bias in estimation. *Biometrika*, 43(3/4), 353–360. https://doi.org/10.1093/biomet/43.3-4.353
- Tukey, J. W. (1958). Bias and confidence in not-quite large samples. *Annals of Mathematical Statistics*, 29(2), 614. (Abstract). https://doi.org/10.1214/aoms/1177706647
- Proschan, F. (1963). Theoretical explanation of observed decreasing failure rate. *Technometrics*, 5(3), 375–383. https://doi.org/10.1080/00401706.1963.10490105
- Bjerkedal, T. (1960). Acquisition of resistance in guinea pigs infected with different doses of virulent tubercle bacilli. *American Journal of Epidemiology*, 72(1), 130–148. https://doi.org/10.1093/oxfordjournals.aje.a120129
- Lee, E. T. and Wang, J. W. (2003). *Statistical Methods for Survival Data Analysis* (3rd ed.). John Wiley & Sons. https://doi.org/10.1002/0471458546
- Meeker, W. Q. and Escobar, L. A. (2021). *Statistical Methods for Reliability Data* (2nd ed.). Wiley Series in Probability and Statistics.
- Zhao, Y., Meng, X., and Yang, H. (2015). Jackknife empirical likelihood inference for the mean absolute deviation. *Computational Statistics and Data Analysis*, 91, 92–101. https://doi.org/10.1016/j.csda.2015.06.001

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{AyeniZhao2026,
  author  = {Ayeni, Taiwo Michael and Zhao, Yichuan},
  title   = {Empirical Likelihood Inference for Skewed Lifetime Models:
             A Jackknife Approach for the {QTEG} Distribution},
  journal = {TBD},
  volume  = {TBD},
  doi     = {TBD},
  note    = {Manuscript in preparation},
  year    = {2026}
}
```

---

## Contact

**Taiwo Michael Ayeni**  (Corresponding Author)
Department of Mathematics and Statistics  
Georgia State University, Atlanta, GA  
Email: tayeni2@gsu.edu

**Yichuan Zhao**   
Department of Mathematics and Statistics  
Georgia State University, Atlanta, GA  
Email: yichuan@gsu.edu

---

## License

This code is made available for academic and research purposes.  
© 2026 Taiwo Michael Ayeni and Yichuan Zhao, Georgia State University.
