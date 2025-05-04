# Forest Cover Type Classification using SVM:
This project focuses on classifying forest cover types using the Forest CoverType dataset from the UCI Machine Learning Repository. The main objective is to build a Support Vector Machine (SVM) classifier capable of predicting the type of forest cover based on cartographic variables.

# Dataset:
The dataset contains cartographic variables such as elevation, aspect, slope, horizontal and vertical distances to hydrology, and soil type indicators
The target variable is the forest cover type (7 classes).

# Preprocessing:
Removed low-variance features to reduce dimensionality and noise.
Normalized continuous features for better SVM performance.
Split the dataset into training and testing sets (70%/30%).

# Model:
Implemented an SVM classifier using scikit-learn's SVC.
Applied GridSearchCV for hyperparameter tuning (e.g., kernel type, regularization parameter C).

# Evaluation:
Evaluated model performance using accuracy score.
Generated confusion matrix to visualize classification performance.
Used seaborn and matplotlib for visualization.

# Technologies Used:
Python (Jupyter Notebook)
scikit-learn
pandas, numpy
seaborn, matplotlib

# Results:
The best model achieved [INSERT FINAL ACCURACY]% accuracy on the test set (update this after running the final model).

Deployment:
The trained model has been deployed as a web application/API for public interaction.



