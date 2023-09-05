import matplotlib.pyplot as plt
import numpy as np
from   scipy.stats import multivariate_normal


x, y = np.mgrid[-5:4:.05,-3:8:.05]
pos  = np.dstack((x, y))
rv   = multivariate_normal([-1, 2], [[2, 1], [1, 4]])

fig, ax = plt.subplots()
CS = ax.contour(x, y, rv.pdf(pos))
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title("$\mu=[-1,2]^T$, $\Sigma=[2,1;1,4]$")
plt.show()