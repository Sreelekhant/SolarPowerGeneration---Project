# 🌞 Solar Power Generation – ML Project 

This project predicts solar power generated based on environmental factors like temperature, humidity, wind speed, and more.

The goal is to build a regression model that learns the relationship between weather conditions and energy production.

We used XGBoost Regressor because it gave the best accuracy among all tested models.

## Poject Files
| File                                   | Description                            |
| -------------------------------------- | -------------------------------------- |
| **SolarPowerGeneration_Updated.ipynb** | Complete EDA + Model Training notebook |
| **solarpowergeneration.csv**           | Dataset                                |
| **app.py**                             | Deployment file (Flask API)            |
| **requirements.txt**                   | All required Python libraries          |

# 🔍 Project Steps 

# 1️⃣ Exploratory Data Analysis (EDA)

Checked missing values

Plotted distributions

Outlier detection

Correlation heatmap

Relationship between features and target

# 2️⃣ Model Building

We tested multiple models:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor (Best Model)

XGBoost gave the highest accuracy and lowest error.

# 3️⃣ Model Saving

The final trained XGBoost model was saved for deployment.

# 4️⃣ Deployment

A simple Flask API was created to load the model and generate predictions based on user inputs.

👉 **Live App:** [Click here to open the Streamlit App 🚀] (https://solarpowergeneration---project-whrtqnwanpawhfeilgyrck.streamlit.app/)

# ✨ Outcome

The project successfully predicts solar power generation using XGBoost with strong performance.
It can be integrated into a web or mobile app to provide real-time energy forecasts.
