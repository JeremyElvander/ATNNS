# A Mixture of Experts Neural Network Framework for Predicting Inorganic Aerosol Thermodynamics
Jeremy M. Elvander and Anthony S. Wexler.

Codebase developed by Jeremy M. Elvander. 
Correspondence to Jeremy Elvander (jelvander@ucdavis.edu) and Anthony Wexler (aswexler@ucdavis.edu).

Repository containing trained models and accompanying scalers, a lightweight class to use the full MoE system, an example notebook on usage, and archive code and data used to:
1. acquire research data
2. process research data
3. train models
4. select models
5. evaluate models

## Purpose

Thermodynamic equilibrium models, such as the Extended Aerosol Inorganics Model (E-AIM), provide accurate predictions of aerosol composition and phase, but are computationally impractical for Eulerian atmospheric models. This work addresses this computational burden by proposing a Mixture of Experts (MoE) neural network framework to efficiently approximate key outputs from E-AIM Model IV, including aerosol water content and partial pressure products for ammonium nitrate and ammonium chloride, which are needed for condensation/evaporation calculations. 

Results from this project highlight how domain-aware routing and physics-informed data transformation within a deep learning framework can enable highly accurate, scalable approximations of complex thermodynamic systems, providing a practical complement to classic equilibrium solvers for large-scale applications.

## Citation

If this repository is used in other research or work, please cite it as:

Elvander, J. and Wexler, A. S.: eaim_moe_neuralnet: A Python Implementation for Inorganic Aerosol Thermodynamic Approximations, Zenodo [code], 10.5281/zenodo.xxxxxxx, 2026.

'''bibtex
@software{elvanderwexler2026eaim,
  author       = {Elvander, J. and Wexler, A. S.},
  title        = {{eaim_moe_neuralnet: A Python Implementation for Inorganic Aerosol Thermodynamic Approximations}},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.xxxxxxx},
  url          = {https://github.com/JeremyElvander/eaim_moe_neuralnet}
}


## Installation

This repository was created with Github Large File Storage (LFS). Downloading this repository may take time. 

To ensure all dependencies are installed, please run pip install -r requirements.txt. Alternatively, manually install dependencies as needed. A virtual environment is recommended. 

## Usage

system.py contains the wrapper class NeuralSystem, which provides an easy interface to access the expert models through the decision-tree framework. example.ipynb includes an implementation example using training data from the project. NeuralSystem.prediction() is the primary method used for predicting thermodynamic outputs.

Repository should be cloned or forked for use. For academic or professional use of this work, please use the appropriate citation. 

### Inputs

NeuralSystem within system.py takes the following inputs for the prediction method:
1. data: n by 7 pandas df with columns [TEMP, RH, NH4+, NA+, SO42-, NO3-, CL-].

### Outputs

The prediction method outputs an n by 4 pandas df of predicted results for [phase, amm_nit, amm_chl, water_content].

## Models

Specific expert models are stored in the models subdirectory. Models are named based off of what leaf node they represent. EX: liqmix_amm_chl_nonzero.keras represents the NH4+ != 0 case, the liquid/mix phase case, ammonium chloride model. Model terminology is as follows:

1. nonzero -> NH4+ != 0 cases
2. zero -> NH4+ = 0 cases
3. liqmix -> liquid/mix phase cases
4. solid -> solid phase cases

## Scalers

Scalers represent statistical scaling for temperature (T) and relative humidity (RH) inputs fir each model. Naming conventions match associated models. Represents a dictionary with 'TEMP' and 'RH' as keys, with a tuple value representing (mean, std).

EX: {'TEMP': [mean, std],
    'RH': [mean, std]
    }

## Data

Example data, and data used for training, is provided in the data subdirectory.

## License

This work is licensed under the Creative Commons Attribution 4.0 International license.


