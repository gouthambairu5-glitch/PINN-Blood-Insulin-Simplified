# System Architecture

## Model
Fully connected neural network:
Input: spatial coordinate (x), time (t)  
Output: velocity u(x, t)

## Physics Loss
Residual of viscous flow PDE:
du/dt − ν d²u/dx²

Derivatives computed using automatic differentiation.

## Training Pipeline
1. Sample collocation points in space–time domain
2. Predict velocity field
3. Compute temporal and spatial derivatives
4. Enforce PDE residual minimization
5. Optimize using Adam

## Output
Velocity heatmap representing blood flow dynamics