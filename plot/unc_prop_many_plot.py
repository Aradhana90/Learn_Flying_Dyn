import joblib
import matplotlib.pyplot as plt
import tikzplotlib

load_path = '../eval/unc_prop_eval_data/Sigma_th_0.1_n_samples_441/'

# Load data
X_rand = joblib.load(load_path + 'X_rand.sav')
n_max = joblib.load(load_path + 'n_max.sav')
good_initial_state = joblib.load(load_path + 'good_initial_state.sav')

# Plot
mask = good_initial_state.astype(bool)
sc = plt.scatter(X_rand[:, 7], X_rand[:, 9], c=n_max, cmap='viridis', marker='.')
plt.scatter(X_rand[mask, 7], X_rand[mask, 9], facecolors='none', edgecolors='black', s=80)
plt.colorbar(sc)
print(n_max)
tikzplotlib.save('./tex_files/unc_prop_many.tex')
plt.show()
