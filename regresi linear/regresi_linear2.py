import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Rounded data points
ages = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Ages (years)
weights = np.array([10, 13, 15, 17, 19])  # Rounded weights

# Create a linear regression model
model = LinearRegression()
model.fit(ages, weights)

# Get the slope (m) and intercept (b)
slope = model.coef_[0]
intercept = model.intercept_

# Predict weights for the ages
predicted_weights = model.predict(ages)

# Print the slope and intercept
print("Slope (m):", slope)
print("Intercept (b):", intercept)

# Plot the data and the regression line
plt.scatter(ages, weights, color="blue", label="Actual Data")
plt.plot(ages, predicted_weights, color="red", label="Regression Line")
plt.xlabel("Age (years)")
plt.ylabel("Weight (kg)")
plt.title("Linear Regression of Age vs. Weight (Rounded Data)")
plt.legend()
plt.show()