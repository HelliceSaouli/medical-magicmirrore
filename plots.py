import numpy as np
import matplotlib.pyplot as plt


x, y = np.loadtxt("plots/TestCase_GT.txt", delimiter=',', unpack=True)
y1 = np.loadtxt("plots/canonic_face_approch.txt", unpack=True)
y2 = np.loadtxt("plots/mediapipe_face_landmark.txt", unpack=True)

plt.plot(x, y, label="GT")
plt.plot(x, y1, label="Approach a")
plt.plot(x, y2, label="Approach b")
plt.xlabel("Frame numbers")
plt.ylabel("Angles in radient")
plt.show()