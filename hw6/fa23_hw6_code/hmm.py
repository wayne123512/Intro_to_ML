#%%
import random
from env import Env
import numpy as np
import matplotlib.pyplot as plt
#%%
def viterbi(observations: list[list[int]], epsilon: float) -> np.ndarray:
    """
    Params: 
    observations: a list of observations of size (T, 4) where T is the number of observations and
    1. observations[t][0] is the reading of the left sensor at timestep t
    2. observations[t][1] is the reading of the right sensor at timestep t
    3. observations[t][2] is the reading of the up sensor at timestep t
    4. observations[t][3] is the reading of the down sensor at timestep t
    epsilon: the probability of a single sensor failing

    Return: a list of predictions for the agent's true hidden states.
    The expected output is a numpy array of shape (T, 2) where 
    1. (predictions[t][0], predictions[t][1]) is the prediction for the state at timestep t
    """
    # TODO: implement the viterbi algorithm
    def calc_emission_prob(truth, obs):
        diff = np.sum(truth != obs)
        return ((1-epsilon) ** (4-diff)) * (epsilon ** diff)
    
    # initialization
    v = np.ones((rows,cols)) * 1.0 / (rows * cols)
    ptr = -1 * np.ones((traj_len, rows, cols, 2), dtype=int)

    for r in range(0, rows):
        for c in range(0, cols):
            truth = env.get_true_sensor_reading(r, c)
            v[r][c] = v[r][c] * calc_emission_prob(truth , observations[0])
    # start the algorithm
    for t in range(1, traj_len):
        v_new = np.zeros((rows,cols))
        for r in range(0,rows):
            for c in range(0, cols):
                tmp_v = 0
                tmp_ptr = [-1,-1] 
                for nghb in env.get_neighbors(r, c):
                    i = nghb[0]
                    j = nghb[1]
                    value = 1.0 / np.size(env.get_neighbors(i, j), axis=0) * v[i][j]
                    if value >= tmp_v:
                        tmp_v = value
                        tmp_ptr = [i, j]
                truth = env.get_true_sensor_reading(r, c)
                v_new[r][c] = tmp_v * calc_emission_prob(truth, observations[t])
                ptr[t][r][c] = np.array(tmp_ptr)
        v = v_new
    r_idx , c_idx = np.unravel_index(v.argmax(), v.shape)
    out = []
    out.append(np.array([r_idx,c_idx]))
    for i in range(1, traj_len):
       r_idx , c_idx = ptr[traj_len-i][r_idx][c_idx]
       out.append(np.array([r_idx,c_idx]))
    
    out.reverse()
    assert len(out) == traj_len, "Mismatch length"
    return np.array(out)

if __name__ == '__main__':
    random.seed(12345)
    rows, cols = 16, 16 # dimensions of the environment
    openness = 0.3 # some hyperparameter defining how "open" an environment is
    traj_len = 100 # number of observations to collect, i.e., number of times to call env.step()
    num_traj = 100 # number of trajectories to run per epsilon

    env = Env(rows, cols, openness)
    env.plot_env() # the environment layout should be saved to env_layout.png

    plt.clf()
    """
    The following loop simulates num_traj trajectories for each value of epsilon.
    Since there are 6 values of epsilon being tried here, a total of 6 * num_traj
    trajectories are generated.
    
    For reference, the staff solution takes < 3 minutes to run.
    """
    for epsilon in [0.0, 0.05, 0.1, 0.2, 0.25, 0.5]:
        env.set_epsilon(epsilon)
        
        accuracies = []
        for _ in range(num_traj):
            env.init_env()

            observations = []
            for i in range(traj_len):
                obs = env.step()
                observations.append(obs)

            predictions = viterbi(observations, epsilon)

            accuracies.append(env.compute_accuracy(predictions))
        plt.plot(np.mean(accuracies, axis=0), label=f"epsilon={epsilon}")

    plt.xlabel("Number of observations")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("accuracies.png")