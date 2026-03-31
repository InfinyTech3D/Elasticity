from pyfreefem import FreeFemRunner
import matplotlib.pyplot as plt
import numpy as np

runner = FreeFemRunner("barre.edp")

exports = runner.execute({
    'L': 1.0,
    'E': 210000.0,
    'A': 0.001,
    'F': 1000.0,
    'N': 10
})

u = np.loadtxt("u.txt")


plt.figure()
plt.plot(u, 'o-')
plt.xlabel("Noeud")
plt.ylabel("Deplacement u(x)")
plt.title("Deplacement de la barre")
plt.grid(True)
plt.show()
