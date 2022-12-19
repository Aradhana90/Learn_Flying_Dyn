# Learn_Flying_Dyn
This project implements Gaussian process (GP) state space models for predicting the flying trajectories of different objects.

The raw measured data, i.e., the measured positions and orientations, is stored in data/raw. 
Mean-filtering and numerical differentiation are applied in data/filter_and_diff.py to create suitable training data.

GP models are trained with cross-validation in eval/train_many.py.

Multiple evaluation scripts are provided in the folder eval/eval_scripts to assess the RMSE, predicted final position and orientation
and the predicted uncertainty.
 