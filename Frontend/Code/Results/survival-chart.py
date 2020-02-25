import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x = 120
y = 200
plt.step(x//365.25, y, where="post", label=str(0))
plt.ylabel("Survival probability")
plt.xlabel("Time (Years)")
plt.title("Overall Survival Curve (in Years)")
plt.grid(True)
plt.legend()

# plt.savefig('assets/kaplan-meier.png',bbox_inches='tight', pad_inches=0)
plt.show()
