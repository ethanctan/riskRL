import matplotlib.pyplot as plt

# Values provided
values = [738, 1738, 2738, 10938, 20848, 21848, 29148, 35728, 44378, 45378, 46378, 47378, 48378, 49378, 50378, 51378, 52378, 53378, 54378, 55378]

# Creating a line graph
plt.figure(figsize=(10, 6))
plt.plot(values, marker='o')
plt.title('Rewards per game in meta-algorithm')
plt.xlabel('Game')
plt.ylabel('Reward')
plt.grid(True)
plt.show()