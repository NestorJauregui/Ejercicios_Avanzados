import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('fake_regression.txt')
x = data[:, 0]
y = data[:, 1]
x_training = x[:70]
y_training = x[:70]
x_test = x[70:]
x_test = x[70:]

coefs = []
for i in range(20):
    #np.append(coefs, np.polyfit(x_training, y_training, i))
    coefs.append(np.polyfit(x_training, y_training, i))

#print np.polyfit(x_training, y_training, 2)

#Seguir aca
