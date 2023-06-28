from random import gauss
import numpy as np
import matplotlib.pyplot as plt

# def gaussian(val, mean, std):
#     return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((val - mean)/ std)**2)

# x_list = np.linspace(0, 1, 100)
# y_list = gaussian(x_list, 0.5, 0.2)

# print(x_list)
# print(y_list)

# plt.plot(x_list, y_list)
# plt.show()

test_list = np.array([[5, 4, 1, 3, 2], [4, 3, 2, 4, 5]])
test_list_2 = ["five", "four", "two", "three", "one"]
# list1, list2 = zip(*sorted(zip(test_list, test_list_2)))
# print(min(test_list[test_list != 1]))
# i, j = np.where(test_list==1)
# print(test_list[i[0], j[0]])
print(np.sum(test_list))
