import numpy as np
import matplotlib.pyplot as plt
em = np.linspace(0.001,0.999,num = 100)
am = 1/2*np.log((1-em)/em)
print(em)
print(am)
y = pow(((1-em)/em),(1/2))
#plt.plot(am, np.exp(am))
plt.plot(em, y)
plt.show()
