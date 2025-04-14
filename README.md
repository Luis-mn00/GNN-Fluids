<div align="center"> 

# MSc Thesis: Physics Informed Artificial Intelligence for Free Surface Fluid Models

*Luis Medrano Navarro, Master in Industrial Mathematics, I3A and UC3M*

This work has been inspired by:

## Graph neural networks informed locally by thermodynamics

*Alicia Tierz, Iciar Alfaro, David González, Francisco Chinesta, and Elías Cueto*

[![Project page](https://img.shields.io/badge/-Project%20page-blue)](https://amb.unizar.es/people/alicia-tierz/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2405.13093)

</div>

## Abstract
This repository contains part of the code used in my MSc thesis at I3A about GNNs for free surface fluid simulations. The goal was to develop a digital twin of sloshing scenarios, where we want to simulate the movement of a fluid inside a container. To have real-time predictions we can not rely on classical numerical solvers, but we use a Deep Learning surrogate model. In particular, we trained a Thermodynamics Informed Graph Neural Network (TIGNN).


## Methodology

### Key Features:
- **Physics-Informed Bias**: Incorporates the GENERIC formalism for thermodynamic consistency.
- **Local Implementation**: Maintains the local structure of GNNs to improve computational scalability.
- **Encode-Process-Decode Architecture**: Implements a multi-layer perceptron-based pipeline for processing nodal and edge features.

### Supported Scenarios:
- Prediction of the initial condition: The final digital twin works with a camera filming the real movement of water inside a glass. In order to start the rollout with the TIGNN we need the initial condition. We can get the initial positions of particles inside the fluid, but the TIGNN also relies on velocity data, which is imposible to get with a camera. Therefore, based on many SPH simulations, we predict the initial velocities with a second GNN.
- Simulation of the fluid: Once we have the initial condition, the TIGNN does the rollout to simulate the temporal evolution in real time.


## Usage

In order to run the training or the inference codes, we first need to gather the data and have the correct folcer structure. For storage purposes, we do not provide the data directly on the repository. Besides the content in the repo, you should create the following folders.
```
GNN_rollout/
├── data_init/
│   ├── dataset/
│   ├── jsonFiles/
│   └── weights/
├── data_rollout/
│   ├── dataset_1/
│   ├── dataset_3/
│   ├── dataset_5/
│   ├── jsonFiles/
│   └── weights/
├── images/
├── outputs_init/
├── outputs_rollout/
```


## Citation

If you use this code or datasets in your research, please cite:

```bibtex
@misc{tierz2024graphneuralnetworksinformed,
      title={Graph neural networks informed locally by thermodynamics}, 
      author={Alicia Tierz and Iciar Alfaro and David González and Francisco Chinesta and Elías Cueto},
      year={2024},
      eprint={2405.13093},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.13093}, 
}
```
### License

This repository is licensed under the GNU License. See `LICENSE` for details.

---

For any questions or feedback, please contact me.