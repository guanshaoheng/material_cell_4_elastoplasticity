# RNN based constitutive model

## RNN trained on J2 plasticity datasets
- model formulas
- random path generation via RGP (random gaussian process)  
    - generate the principal stress and Lode angle
    - revert to the tensor components
    - strain initial [0, 0]
    - principal strain max [0.1, 0.18]
    - Lode angle initial np.random.uniform([-pi, pi])
    - lode angle max [0, pi/6]
- J2 model loading simulation and results save
- training based on PyTorch
  - num_layers = 2
  - parameters  8943
  - num_steps = 200
  - num_inputs = 3
  - num_outputs = 3 (currently)

