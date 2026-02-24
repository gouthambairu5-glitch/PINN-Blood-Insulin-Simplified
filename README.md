# PINN for Blood Flow Hemodynamics (Cardiovascular Modeling)

## Overview
Physics-Informed Neural Network for modeling simplified 1D blood flow dynamics using a viscous flow equation.

## Medical Relevance
Hemodynamic modeling is used in cardiovascular research for studying arterial flow and vascular resistance.

## Governing Equation
du/dt = ν d²u/dx²

Where:
- u → Blood velocity
- ν → Kinematic viscosity

## Method
The neural network learns the velocity field by minimizing PDE residual loss computed via automatic differentiation.

## Features
- PDE-constrained learning
- Spatiotemporal velocity prediction
- Heatmap visualization of flow field

## Run
python train.py