This project implements a machine learning pipeline to classify astronomical objects—Galaxy, Quasar, and Star—using data from the Sloan Digital Sky Survey (SDSS). The classification is performed using Decision Trees, Logistic Regression, and K-Nearest Neighbors (KNN). The objective is to identify the object type based on photometric and spectroscopic features.


Dataset
The dataset used in this project is derived from the Sloan Digital Sky Survey (SDSS). It includes various photometric and spectroscopic features and a target column specifying the object class
Steps performed on the dataset:
* Loaded the dataset and inspected its structure.
* Removed irrelevant columns such as objid and specobjid.
* Encoded the target labels using LabelEncoder (0 = Galaxy, 1 = Quasar, 2 = Star).
* Scaled the feature columns using StandardScaler to normalize the data.
* Split the dataset into training (70%) and testing (30%) subsets.
Algorithms Used
1. Decision Tree Classifier:
   * Hyperparameters: max_leaf_nodes=15, max_depth=3
   * A tree-based algorithm to create decision rules for classification.
2. Logistic Regression:
   * A linear model suitable for binary or multiclass classification.
3. K-Nearest Neighbors (KNN):
   * Hyperparameters: n_neighbors=3
   * A non-parametric method that classifies a data point based on its nearest neighbors.
Metrics for Evaluation
* Classification report including:
   * Precision
   * Recall
   * F1-Score
   * Accuracy


Implementation
Libraries Used
* Data Handling and Manipulation: numpy, pandas
* Data Visualization: matplotlib, seaborn
* Modeling and Evaluation: scikit-learn, tensorflow
Pipeline
1. Data Preprocessing:
   * Dropped irrelevant columns.
   * Encoded the target variable.
   * Normalized feature columns using StandardScaler.
2. Data Visualization:
   * Used seaborn to visualize class distributions and feature dependencies (e.g., count plots, pair plots).
3. Model Training:
   * Implemented three models: Decision Tree, Logistic Regression, and KNN.
   * Trained each model on the training dataset.
4. Model Evaluation:
   * Predicted classes for the test dataset using each model.
   * Evaluated predictions using classification metrics.


Results
* Decision Tree:
   * Achieved an interpretable set of decision rules but may have limitations due to shallow tree depth.
* Logistic Regression:
   * Demonstrated robust linear classification performance.
* K-Nearest Neighbors:
   * Achieved high accuracy due to its local decision-making process.
Sample outputs of the predictions for the first 10 test samples are displayed for each model, along with detailed classification reports.
Future Scope
* Experiment with additional models like SVM, Random Forests, or Neural Networks.
* Fine-tune hyperparameters to optimize model performance.
* Extend the dataset to include more features and classes for comprehensive analysis.
* Create a web application for interactive classification of astronomical objects.
