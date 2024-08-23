import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

plt.figure()
plt.text(0.5, 0.5, r"$E=mc^2$", fontsize=20)
plt.show()
