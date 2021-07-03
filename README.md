This case study is based on very famous Dataset on Kaggle in Machine Learning.The Bank-Note Authentication Data.
The data contains information about bankNote is authentic given a number of measures taken. The dataset contains 1,372 rows and 5 columns
numeric variables.It is a Classification problem with two binary classes.
The flow of the case study is as below:
Reading the data in python
Defining the problem statement
Identifying the Target variable
Looking at the distribution of Target variable
Basic Data exploration
Rejecting useless columns
Visual Exploratory Data Analysis for data distribution (Histogram and Barcharts)
Feature Selection based on data distribution
Outlier treatment
Missing Values treatment
Visual correlation analysis
Statistical correlation analysis (Feature Selection)
Converting data to numeric for ML
Sampling and K-fold cross validation
Trying multiple classification algorithms
Selecting the best Model
Deploying the best model in production
I have used Alogorithms to deploy models are:
1.Logistic Regression Classififcation-Report
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       220
           1       1.00      1.00      1.00       185

    accuracy                           1.00       405
   macro avg       1.00      1.00      1.00       405
weighted avg       1.00      1.00      1.00       405

Confusion-Matrix
 [[220   0]
 [  0 185]]
Accuracy of the model on Testing Sample Data: 1.0

Accuracy values for 10-fold Cross Validation:
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

Final Average Accuracy of the model: 1.0
2.K-Nearest Neighbor(KNN) KNeighborsClassifier(n_neighbors=3)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       220
           1       1.00      1.00      1.00       185

    accuracy                           1.00       405
   macro avg       1.00      1.00      1.00       405
weighted avg       1.00      1.00      1.00       405

[[220   0]
 [  0 185]]
Accuracy of the model on Testing Sample Data: 1.0

Accuracy values for 10-fold Cross Validation:
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

Final Average Accuracy of the model: 1.0

3. Deep Neural Network (ANN) Artificial Neural Network:
BinaryLayer model is a class of Artificial Neural Network that uses back Propagation technique for training.It has two layers of Nodes.
The hidden layer has 2, input are 4 and used activation Rectifier Function.
The Outer layer contains 1 unit and Sigmoid Activation function and optimizer adam used. Batch_Size is 10 and epochs are 10 and get 
the last accuracy value of epochs as 1.0.
To find the best set of parameters used grid search where epochs is 100 and parameters passed is 16
1 Parameters: batch_size: 5 - epochs: 5 Accuracy: 1.0
2 Parameters: batch_size: 5 - epochs: 10 Accuracy: 1.0
3 Parameters: batch_size: 5 - epochs: 50 Accuracy: 1.0
4 Parameters: batch_size: 5 - epochs: 100 Accuracy: 1.0
5 Parameters: batch_size: 10 - epochs: 5 Accuracy: 1.0
6 Parameters: batch_size: 10 - epochs: 10 Accuracy: 1.0
7 Parameters: batch_size: 10 - epochs: 50 Accuracy: 1.0
8 Parameters: batch_size: 10 - epochs: 100 Accuracy: 1.0
9 Parameters: batch_size: 15 - epochs: 5 Accuracy: 1.0
10 Parameters: batch_size: 15 - epochs: 10 Accuracy: 1.0
11 Parameters: batch_size: 15 - epochs: 50 Accuracy: 1.0
12 Parameters: batch_size: 15 - epochs: 100 Accuracy: 1.0
13 Parameters: batch_size: 20 - epochs: 5 Accuracy: 1.0
14 Parameters: batch_size: 20 - epochs: 10 Accuracy: 1.0
15 Parameters: batch_size: 20 - epochs: 50 Accuracy: 1.0
16 Parameters: batch_size: 20 - epochs: 100 Accuracy: 1.0
After finding the parameters with the epoches Generating the predictors on Testing Data getting the testing data values
	variance	skewness	curtosis	class	PredictedBankNoteProb	PredictedBankNote
0	0.051979	7.052100	-2.05410	0	3.072792e-09	0
1	-0.942550	0.039307	-0.24192	1	9.999920e-01	1
2	-2.826700	-9.040700	9.06940	1	9.999920e-01	1
3	-3.885800	-12.846100	12.79570	1	9.999920e-01	1
4	-1.804600	-6.814100	6.70190	1	9.999920e-01	1

Calculating the accuracy on testing data
Classification Report
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       221
           1       1.00      1.00      1.00       184

    accuracy                           1.00       405
   macro avg       1.00      1.00      1.00       405
weighted avg       1.00      1.00      1.00       405

Confusion Matrix
 [[221   0]
 [  0 184]]

