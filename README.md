# Learn_Flying_Dyn
This project aims at learning the dynamics of free-flying objects from recorded real-world trajectories. For this purpose, Gaussian process regression (GPR) and support vector regression (SVR) are employed. The system dynamics can be learned either in continuous-time (mapping from 13D state to 6D acceleration) or discrete-time (mapping from 13D current state to 13D next state). 

# Data generation
1. Store rosbag files in data/raw
2. Convert to csv by running data/rosbag2csv.py
3. Apply filtering and numerical differentiation with the functions in data/filter_and_diff.py. This is done automatically by the Data_Handler class.

# Training
1. Run eval/train_many.py. Here, you can select SVR or GPR, continuous-time or discrete-time, the number of training trajectories, the number of cross-validations and the used kernel function. The trained models are then stored to be later used for evaluation.

# Evaluation
There are different scripts to evaluate the performance using different metrics. In the scripts, you can select between GPR/SVR and CT/DT.
- RMSE: Run eval/eval_scripts/eval_rmse.py
- Final position and orientation error: Run eval/eval_scripts/eval_traj.py
- Uncertainty propagation: Run eval/eval_scripts/uncertainty_propagation.py

Shown below is the deviation of the final position and orientation for the continuous-time prediction model for different numbers of training trajectories.
![plot](/plot/pred_error_cont.JPG)

Shown below is the result of approximate uncertainty propagation with the discrete-time prediction model and GPR.
![plot](/plot/unc_prop.JPG)

Shown below is the result of applying approximate uncertainty propagation with many different initial throwing velocities. This is done in order to obtain the most reliable throwing configuration.
![plot](/plot/configurations_uncertainty.JPG)

# Other 
- The learning algorithms are implemented as classes in learning/gpr and learning/svr. The uncertainty propagation scheme is implemented in learning/gpr/GPR_disc.py.
Various scripts for plotting are available in the directory plot. 
- Different evaluation metrics such as RMSE and the deviation in final position and orientation are implemented in eval/functions/metrics.py.
