# Posterior Estimation of Dynamic PET Kinetic Parameters with Improved DDPM



Reference implementation of the **improved Denoising Diffusion Probabilistic Model (iDDPM)** described in our IEEE TMI 2025 paper:

> **Djebra Y., Liu X., Marin T. *****et al.***  \
> *Bayesian Posterior Distribution Estimation of Kinetic Parameters in Dynamic Brain PET Using Generative Deep‑Learning Models*  \
> IEEE Transactions on Medical Imaging, 2025  \
> DOI 10.1109/TMI.2025.3588859

The model learns the *full Bayesian posterior* over Distribution Volume Ratio (**DVR**) and relative delivery rate constant (**R₁**) directly from noisy time–activity curves (TACs). Compared with a Metropolis‑Hastings MCMC baseline it achieves > 230× faster inference while keeping the absolute error < 0.67 % (mean) and < 7.3 % (SD).

---

## Repository layout

```
.
├── diffusion_model.py       # iDDPM implementation (Keras/TensorFlow)
├── networks.py              # U‑Net + conditioning encoder
├── helper_func.py           # utilities (priors, beta schedules, math)
├── kinetic_model.py         # SRTM / SRTM2 forward models
├── sample_sim_data.py       # generate synthetic TACs (training / test)
├── main_script.py           # end‑to‑end training script
├── mcmc.py                  # MCMC posterior sampler for benchmarking
├── requirements.txt         # Python dependencies (TF 2.10, TensorFlow‑Prob, PyMC…)
└── docs/
    └── Djebra2025_accepted_manuscript.pdf
```

*(Heavy checkpoints live under **`results/`** once you train; they’re not needed to run the code.)*

---

## Installation

```bash
# clone
$ git clone https://github.com/yanisdjebra/PET_posterior_distribution.git
$ cd PET_posterior_distribution

# Python 3.9+ environment (GPU strongly recommended)
(See requirements.txt)
```

> ✅ The repo was tested with **TensorFlow 2.10 + CUDA 11.8** on Linux and WSL2.

---

## 1st step: Data preparation

Synthetic TACs can be generated from SRTM2 with realistic priors learned from 50 [¹⁸F]MK‑6240 scans.
Run `sample_sim_data.py`, changing accordingly the simulation parameters and directories:
```
HOME_DIR, n_samples, n_ROI, flag_testing_data, mean_sigma_noise_save, alpha
```

The script saves `data_nROI48_n100000_s1.0e-01.pik` (train) and a test set under `sim_data/`.

If you already have real TACs, adapt `sample_sim_data.py` to point to your prior calculated from your ROI‑wise measurements.

---

## Training

`main_script.py` trains an iDDPM with a 1‑D U‑Net backbone and saves checkpoints every 50 epochs:

```bash
$ python main_script.py
```
`HOME_DIR` is defined as the root of the git repository. Parameters to load the simulated TACs `(n_samples_orig, n_samples_test, n_ROI, mean_sigma_noise_save)` are defined in the `Load TACs` section for training and testing datasets, automatically loading the data from the directory created in `sample_sim_data.py`.

Parameters for the U-net can be changed in the `Parameters for neural network` section. `net_args` defines the network to use, e.g., the number of initial filters, depth, kernel size, etc—**edit them as needed**.

Training parameters (epochs, learning rate schedule, time steps, noise schedule for diffusion) are also set in this section—**edit them as needed**. By default it trains for **500 epochs** on 100 k simulated TACs and DVR/R1.

Checkpoints, loss curves and posterior plots land in

```
results/nROI48/<time>_<model_name>_f128_d4_p2_s1.0e-01/
```

---

## Posterior inference and figure plots

After training you can draw posterior samples with the stored model at each checkpoint. This is performed in the `Load trained model and infer neural network` section. Parameters for inference and plots are defined above in the `Params for plot` section. For example, the user can modify the number of posterior sample to be generated (`n_posterior`), for which training epoch to load the model for inference (`make_png_period`), if one wants to plot figures (`FLAG_PLOT`), etc.

---

## MCMC baseline

To reproduce the Metropolis‑Hastings ground truth used in the paper, run in python mcmc.py

The script saves per‑ROI chains and diagnostics in `sim_data/` and plots some figures showing the posterior distribution of kinetic parameters.


## Results

After training, `main_script.py` outputs figures in png, pdf and eps format in the save directory (in results folder) such as a bar-plot of posterior distribution using the proposed method vs MCMC:

md\n![Bar plot](./results/nROI48/25-07-10_16-13-15_train/25-07-14_16-35_improved_ddpm_f128_d4_p2_s1.0e-01/cp_450/_s1.0e-01/barplot_epoch450_km_obs-0.842-0.833-0.013.png)\n\n*Bar plot representing the posterior distribution of kinetic parameters DVR (top) and R1 (bottom) per ROI for a testing measurement.*\n

To show the diffusion process, one can use the flag `keep_all_xt=True` in the `Params for plot` section and create a movie of the diffusion process, i.e. from noise to the approximated posterior distribution p(x|y). In such case, and if `FLAG_PLOT` is True, ``main_script.py`` will create the following GIF and MP4 file:

md\n![Movie DVR](./results/nROI48/25-07-10_16-13-15_train/25-07-14_16-35_improved_ddpm_f128_d4_p2_s1.0e-01/cp_450/_s1.0e-01/dist_DVR_epoch450_km_obs-0.842-0.833-0.013.gif)\n\n*Movie of the diffusion process of trained iDDPM, from noise to p(x|y) for DVR*\n

md\n![Movie R1](./results/nROI48/25-07-10_16-13-15_train/25-07-14_16-35_improved_ddpm_f128_d4_p2_s1.0e-01/cp_450/_s1.0e-01/dist_R1_epoch450_km_obs-0.842-0.833-0.013.gif)\n\n*Movie of the diffusion process of trained iDDPM, from noise to p(x|y) for R1*\n

## Citation

```bibtex
@ARTICLE{Djebra2025,
  author={Djebra, Yanis and Liu, Xiaofeng and Marin, Thibault and Tiss, Amal and Dhaynaut, Maeva and Guehl, Nicolas and Johnson, Keith and Fakhri, Georges El and Ma, Chao and Ouyang, Jinsong},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Bayesian Posterior Distribution Estimation of Kinetic Parameters in Dynamic Brain PET Using Generative Deep Learning Models}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2025.3588859}}
```

---

## License & acknowledgements

This code is released for **academic research** under the MIT license (see `LICENSE`).

Portions of the training pipeline were adapted from [Ho *et al.* (NeurIPS 2020)] Denoising Diffusion Probabilistic Models and [Nichol & Dhariwal (2021)] Improved DDPM.

Research supported by NIH grants **P41EB022544, R01EB033582, R01EB035093, R21EB034911, R01AG076153, R21AG070714, R01AG085561,** and **P01AG036694**.

