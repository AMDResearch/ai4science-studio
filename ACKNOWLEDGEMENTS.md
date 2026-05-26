# Acknowledgements

AI4Science Studio curates recipes for open AI-for-science models. The models themselves, their weights, and the ideas behind them are the work of the original authors listed below. AMD Silo AI contributed ROCm/AMD-specific recipes and blog posts for many of these models.

---

## Earth Science

### StormCast
- **Upstream:** NVIDIA Earth-2 Studio — [`NVIDIA/earth2studio`](https://github.com/NVIDIA/earth2studio)
- **Paper:** Bodnar et al., *Kilometer-Scale Convection Allowing Model Emulation using Generative Diffusion Modeling*, arXiv:2408.10958
- **ROCm blog (inference):** [Running StormCast on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/stormcast-inference/README.html) — Pauli Pihajoki (AMD Silo AI)
- **ROCm blog (ensembles):** [StormCast Ensemble Forecasting on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/stormcast-ensembles/README.html) — Pauli Pihajoki (AMD Silo AI)

### ORBIT-2
- **Upstream:** [`XiaoWang-Github/ORBIT-2`](https://github.com/XiaoWang-Github/ORBIT-2); weights on [Hugging Face](https://huggingface.co/jychoi-hpc/ORBIT-2)
- **Paper:** Wang et al., *ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling*, arXiv:2505.04802
- **Dataset DOI:** [10.13139/OLCF/2589526](https://doi.org/10.13139/OLCF/2589526)
- **Maintained by:** Oak Ridge National Laboratory (ORNL)

### ArchesWeather
- **Upstream:** [`gcouairon/ArchesWeather`](https://huggingface.co/gcouairon/ArchesWeather); recipe code from [`silogen/ai-samples`](https://github.com/silogen/ai-samples)
- **Paper:** Couairon et al., *ArchesWeather & ArchesWeatherGen: efficient AI weather forecasting*, arXiv:2412.12971
- **ROCm blog:** [Training ArchesWeather on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/geoarches-training/README.html) — Luka Tsabadze, Rahul Biswas, Pauli Pihajoki, Daniel Warna, Baiqiang Xia, Sopiko Kurdadze (AMD Silo AI)

### Aurora
- **Upstream:** [`microsoft/aurora`](https://github.com/microsoft/aurora); recipe code from [`silogen/ai-samples`](https://github.com/silogen/ai-samples)
- **Paper:** Bodnar et al., *A foundation model of the Earth system*, Nature 2025, https://doi.org/10.1038/s41586-025-08897-0
- **ROCm blog:** [Running SOTA AI-based Weather Forecasting models on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html) — Luka Tsabadze, Rahul Biswas, Pauli Pihajoki, Daniel Warna, Baiqiang Xia (AMD Silo AI)

### GenCast
- **Upstream:** [`google-deepmind/graphcast`](https://github.com/google-deepmind/graphcast); recipe code from [`silogen/ai-samples`](https://github.com/silogen/ai-samples)
- **Paper:** Price et al., *GenCast: Diffusion-based ensemble weather forecasting for improved prediction accuracy and uncertainty quantification*, Nature 2025, https://doi.org/10.1038/s41586-024-08252-9
- **ROCm blog:** [Running SOTA AI-based Weather Forecasting models on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html) — Luka Tsabadze, Rahul Biswas, Pauli Pihajoki, Daniel Warna, Baiqiang Xia (AMD Silo AI)

### NeuralGCM
- **Upstream:** [`google-research/neuralgcm`](https://github.com/google-research/neuralgcm); weights via Google Cloud Storage
- **Papers:**
  - Kochkov et al., *Neural General Circulation Models for Weather and Climate*, arXiv:2311.07222
  - Kochkov et al., *Neural GCMs optimized to predict satellite-based precipitation*, arXiv:2412.11973

### PanguWeather
- **Upstream:** [`198808xc/Pangu-Weather`](https://github.com/198808xc/Pangu-Weather); recipe code from [`silogen/ai-samples`](https://github.com/silogen/ai-samples)
- **Paper:** Bi et al., *Accurate medium-range global weather forecasting with 3D neural networks*, Nature 2023, https://doi.org/10.1038/s41586-023-06185-3
- **ROCm blog:** [Running SOTA AI-based Weather Forecasting models on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/ai-weather-forecasting/README.html) — Luka Tsabadze, Rahul Biswas, Pauli Pihajoki, Daniel Warna, Baiqiang Xia (AMD Silo AI)

---

## Material Science

### MatterGen
- **Upstream:** [`microsoft/mattergen`](https://github.com/microsoft/mattergen)
- **Paper:** Zeni et al., *MatterGen: a generative model for inorganic materials design*, Nature 2025, https://doi.org/10.1038/s41586-025-08628-5
- **ROCm blog:** [Running MatterGen on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/mattergen/README.html) — Sopiko Kurdadze (AMD Silo AI)

### HydraGNN
- **Upstream:** [`ORNL/HydraGNN`](https://github.com/ORNL/HydraGNN) (branch `Predictive_GFM_2024`); weights on [Hugging Face](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024)
- **Dataset DOI:** [10.13139/OLCF/2474799](https://doi.org/10.13139/OLCF/2474799)
- **Citation:** M. Lupo Pasini et al., *HydraGNN_Predictive_GFM_2024 — Ensemble of predictive graph foundation models for group state atomistic materials modeling*, DOI 10.13139/OLCF/2474799
- **Maintained by:** Oak Ridge National Laboratory (ORNL)

---

## Healthcare & Life Sciences

### GP-MoLFormer
- **Upstream:** [`IBM/gp-molformer`](https://github.com/IBM/gp-molformer)
- **Paper:** Ross et al., *Generative Pre-trained Transformer for De Novo Drug Design and Molecular Property Optimization*, arXiv:2302.07432
- **ROCm blog:** [Running GP-MoLFormer on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/gp-molformer/README.html) — Sopiko Kurdadze (AMD Silo AI)

### SwinUNETR
- **Upstream:** [`Project-MONAI/research-contributions`](https://github.com/Project-MONAI/research-contributions); recipe code from [`silogen/ai-samples`](https://github.com/silogen/ai-samples)
- **Paper:** Hatamizadeh et al., *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images*, arXiv:2201.01266
- **ROCm blog (training):** [Running SwinUNETR on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/running-swinunetr-amd/README.html) — Joaquin Rives Gambin (AMD Silo AI)
- **ROCm blog (inference optimization):** [SwinUNETR Inference Optimization on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/swinunetr-inference-optimization/README.html) — Joaquin Rives Gambin, Vasumathi Neralla, David Björelind, Rui Sampaio (AMD Silo AI / AstraZeneca × AMD collaboration)
- **AstraZeneca × AMD collaboration:** https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html

### SemlaFlow
- **Upstream:** [`rssrwn/semla-flow`](https://github.com/rssrwn/semla-flow)
- **Paper:** Morehead et al., *SemlaFlow: Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching*, OpenReview: https://openreview.net/forum?id=bee2G6pEh0
- **ROCm blog:** [Running SemlaFlow on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/semlaflow/README.html) — Vasumathi Neralla, Rui Sampaio (AMD Silo AI / AstraZeneca × AMD collaboration)
- **AstraZeneca × AMD collaboration:** https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html

### REINVENT4
- **Upstream:** [`MolecularAI/REINVENT4`](https://github.com/MolecularAI/REINVENT4)
- **Paper:** Loeffler et al., *REINVENT4: Modern AI-driven generative molecule design*, J Cheminformatics 2024, https://doi.org/10.1186/s13321-024-00812-5
- **ROCm blog:** [Running REINVENT4 on AMD Instinct](https://rocm.blogs.amd.com/artificial-intelligence/running-reinvent4-amd/README.html) — David Björelind, Rui Sampaio (AMD Silo AI / AstraZeneca × AMD collaboration)
- **AstraZeneca × AMD collaboration:** https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html

---

## Physics Simulation

### MATEY
- **Upstream:** [`ORNL/MATEY`](https://github.com/ORNL/MATEY)
- **Paper:** Subramanian et al., *MATEY: multiscale adaptive foundation models for spatiotemporal physical systems*, arXiv:2412.20601
- **Maintained by:** Oak Ridge National Laboratory (ORNL)

### Walrus
- **Upstream:** [`PolymathicAI/walrus`](https://github.com/PolymathicAI/walrus); weights on [Hugging Face](https://huggingface.co/polymathic-ai/walrus)
- **Paper:** McCabe et al., *Walrus: A Cross-Domain Foundation Model for Continuum Dynamics*, arXiv:2511.15684
- **Maintained by:** Polymathic AI

---

## AMD Silo AI

ROCm-specific recipes for many models in this repository were authored by the [AMD Silo AI](https://www.amd.com/en/solutions/ai/silo-ai.html) team and published on the [ROCm Blogs](https://rocm.blogs.amd.com) platform. Contributors across these recipes include: Pauli Pihajoki, Luka Tsabadze, Rahul Biswas, Sopiko Kurdadze, Daniel Warna, Baiqiang Xia, Joaquin Rives Gambin, Vasumathi Neralla, David Björelind, and Rui Sampaio.

## AstraZeneca × AMD Collaboration

The recipes for REINVENT4, SemlaFlow, and the SwinUNETR inference optimization were developed as part of a collaboration between AstraZeneca and AMD. See the [collaboration blog post](https://www.amd.com/en/blogs/2025/astrazeneca-improved-life-sciences-model-training-time.html) for background. The SwinUNETR training recipe and GP-MoLFormer are AMD Silo AI contributions independent of this collaboration.
