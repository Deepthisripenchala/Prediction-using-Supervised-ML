import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)  # Load your dataset here
print("Data Imported successfully")
data.head(10)
X = data.iloc[:, :-1].values  # Independent variable (Hours)
y = data.iloc[:, 1].values    # Dependent variable (Scores)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title('Actual vs. Predicted Scores')
plt.xlabel('Hours of Study')
plt.ylabel('Percentage Score')
plt.show()
new_hours = 9.25  # Example: Predicting the score for 9.25 hours of study
predicted_score = regressor.predict(np.array([[new_hours]]))
print("Predicted Score for {} hours of study: {:.2f}%".format(new_hours, predicted_score[0]))



