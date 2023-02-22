# Learn_Flying_Dyn
This project aims at learning the dynamics of free-flying objects from recorded real-world trajectories. For this purpose, Gaussian process regression (GPR) and support vector regression (SVR) are employed. The system dynamics can be learned either in continuous-time (mapping from 13D state to 6D acceleration) or discrete-time (mapping from 13D current state to 13D next state). 

# Data generation
1. Store rosbag files in data/raw
2. Convert to csv by running data/rosbag2csv.py
3. Apply filtering and numerical differentiation with the functions in data/filter_and_diff.py. This is done automatically by the Data_Handler class.

# Training
1. Run eval/train_many.py. Here, you can select SVR or GPR, continuous-time or discrete-time, the number of training trajectories, the number of cross-validations and the used kernel function.

# Evaluation
1. Run either eval/eval_cont.py or eval/eval_discrete.py depending on whether the continuous-time or discrete-time case is considered. 

# Other 
The learning algorithms are implemented as classes in learning/gpr and learning/svr. The uncertainty propagation scheme is implemented in learning/gpr/GPR_disc.py.
Various scripts for plotting are available in the directory plot. 
