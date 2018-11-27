# ML for Dummies
Run a bunch of machine learning algorithms on any CSV file without any coding. Here's a [demo](https://www.youtube.com/watch?v=Ip2sFECF_is&t=125s) of an older version.

## Dependencies Required
* [Anaconda](https://www.anaconda.com/download/#macos)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/build.html)

## Fields
* **Target class**: This is the class that you want to predict. This needs to specified before opening your dataset.
*  **Null % at which columns are dropped**: If you enter a value of say, 60 here and a column in your dataset has 60% of its values missing, then it will be dropped automatically. Because it is better to drop that column entirely than drop 60% of your dataset’s rows.
* **Max label count for One Hot Encoding**: This determines the threshold for one hot encoding. Suppose you enter a value of 3 and a column (Gender) in your dataset has three unique values (Male, Female, Other), then that column will be one hot encoded. Also, any column (e.g: Type) with less than three unique values (e.g: Free, Paid) will be one hot encoded. All other columns (with unique values > threshold set) will be label encoded.
* **Select algorithm**
* **See classification report**: If selected No, only the accuracy/error of the algorithm will be displayed in the main window after running the program. Otherwise, a new window containing the following information will be opened:
    *  Classifier accuracy
    * Null accuracy: This is the accuracy you would obtain if you always predicted the most frequently occurred value in your target class. Suppose your dataset has 100 rows and your target class has two unique labels Free and Paid, occupying 70 and 30 rows respectively. If you always predict Free, you will be right 70/100 times.
    * Cross validated: Whether 10 fold cross validation was used to evaluate your model or not
    * Confusion matrix
    * Classification report containing precision, recall, f1-score and support of your model
    * Time taken
* **Perform cross validation**: Yes means 10 fold cross validation is used to evaluate your model. No means a train - test ratio of 67:33 is used instead.
* **Try every possible feature combination**: This runs your chosen algorithm on ALL possible combinations of the columns in your dataset. For instance, if your dataset has three columns - A,B and C, then the algorithm will be executed 7 times on the following combinations (A,B,C), (A,B), (A,C), (B,C), (A), (B), (C). Note that the live results for each iteration will be displayed in the console of your IDE which includes:
    * Best score and corresponding feature combination
    * Current score and corresponding feature combination
* **Drop columns manually**: Desired outcome needs to be selected before opening file. 
* **Feature selector**: Finds the most important features in your dataset

## Buttons
* **Open File**: Only CSV files are supported.
* **Run**: The chosen algorithm is executed on the dataset opened.
* **Export CSV**: Generates an encoded version of the CSV (based on the values of the first three fields) in the directory of the original file.

## Data Summary
This option appears once a file has been opened. It contains the following information of your dataset:
* **Total column count**
* **Total row count**
* **Null row count**: The number of rows with a missing value in at least one corresponding cell
* **All column names** followed by:
    * The percentage of missing values
    * Number of unique values
* **Bar chart**: Generates an ordered bar chart of the chosen column
* **Histogram**: Only numerical options are shown. Hence, it may be a useful tool to identify which columns will need encoding.
* **Scatter Plot**: Allows visualisation of the relationship between any two numeric columns
