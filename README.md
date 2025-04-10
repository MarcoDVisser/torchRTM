# TorchRTM: A PyTorch-based Radiative Transfer Model

**TorchRTM** is a flexible and efficient package designed for simulating radiative transfer in the atmosphere, vegetation, and soil systems, based on the **PROSAIL** model. It leverages GPU-accelerated computations and integrates various atmospheric correction models, including **SMAC** for top-of-atmosphere reflectance estimation. This package is particularly suitable for remote sensing and environmental monitoring applications and is build for pyTorch integration.

---

## Features

- **GPU Acceleration**: Built with PyTorch in mind, TorchRTM offers GPU inference to speed up simulations.
- **SMAC Integration**: Perform atmospheric correction using the Simplified Model for Atmospheric Correction (SMAC).
- **PROSAIL Model**: Integrates the PROSAIL radiative transfer model for canopy reflectance simulations.
- **Flexible Inputs**: Handles both batch and individual simulations of canopy and atmospheric parameters.

---

## Installation

### Prerequisites
Ensure that you have Python 3.6+ and PyTorch installed. If you need PyTorch, follow the installation instructions from the [official site](https://pytorch.org/get-started/locally/).

### Install TorchRTM

You can install **TorchRTM** directly from GitHub:

```bash
pip install git+https://github.com/MarcoDVisser/torchrtm.git
```

---

## Usage

### 1. Simulating Canopy Reflectance

The `prosail_shell_v2` function can be used to simulate the canopy reflectance based on input vegetation parameters.

```python
from torchrtm.models import prosail_shell_v2

# Simulate canopy reflectance
toc_full = prosail_shell_v2(
    traits=torch.tensor([[40.0, 8.0, 0.0, 0.01, 0.008]]),  # Example vegetation traits
    N=torch.tensor([1.5]),
    LIDFa=torch.tensor([[-0.3]]),
    LIDFb=torch.tensor([[-0.1]]),
    lai=torch.tensor([3.0]),
    q=torch.tensor([0.5]),
    tts=torch.tensor([30.0]),
    tto=torch.tensor([20.0]),
    psi=torch.tensor([10.0]),
    tran_alpha=torch.tensor([40.0]),
    psoil=torch.tensor([0.5]),
    batch_size=1,
    use_prospectd=False,
    lidtype=1
)
```

### 2. Atmospheric Correction with SMAC

Use the `smac` function to apply atmospheric correction using the SMAC model.

```python
from torchrtm.atmosphere.smac import smac
from torchrtm.data_loader import load_smac_sensor

# Load sensor-specific coefficients
sensor_name = "Sentinel2A-MSI"
coefs = load_smac_sensor(sensor_name)

# Set atmospheric parameters
tts = torch.tensor([30.0])
tto = torch.tensor([20.0])
psi = torch.tensor([10.0])

# Apply SMAC atmospheric correction
ttetas, ttetav, tg, s, atm_ref, tdir_tts, tdif_tts, tdir_ttv, tdif_ttv = smac(tts, tto, psi, coefs)
```

### 3. Converting TOC to TOA Reflectance

Use the `toc_to_toa` function to convert Top-of-Canopy (TOC) reflectance to Top-of-Atmosphere (TOA) reflectance.

```python
from torchrtm.atmosphere.smac import toc_to_toa

# Convert TOC to TOA reflectance
R_TOC, R_TOA = toc_to_toa(toc, sm_wl, ta_ss, ta_sd, ta_oo, ta_do, ra_so, ra_dd, T_g, return_toc=True)
```

---

## Tests

This package includes a set of unit tests to ensure the correctness of all key components. To run the tests, use pytest:

```bash
pytest
```

---

## Contributing

Contributions to **TorchRTM** are welcome! Please fork this repository, create a new branch, and submit a pull request. Make sure to include tests for any new features or bug fixes.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

To be added.
