import random
from env import Env
import numpy as np
import matplotlib.pyplot as plt

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
    pass
    

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