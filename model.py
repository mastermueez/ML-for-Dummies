#Model: Data Structure.
#   --Controller can send messages to it, and model can respond to message.
#   --Uses delegates from vc to send messages to the Controll of internal change
#   --NEVER communicates with View
#   --Has setters and getters to communicate with Controller

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier




from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, GenericUnivariateSelect
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import operator

import itertools
import os
import time
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Model():

    nullPercentageAtWhichColumnsAreDropped = None
    maxLabelCountWhereOneHotEncodePerformed = None
    fileNameAndPath = None
    className = None
    algorithmChosen = None
    dropColumnsManually = False
    performCrossValidation = None
    seeClfReport = None
    dropColumnsManually = None
    userDefinedColumnsToBeDropped = None
    alternateEveryFeature = None
    findBestHyperParameters = None
    
    featureColumnList = None
    strBestFeatures = None
    strCurrentFeatures = None
    df = None
    original_df = None
    score = None

    def __init__(self,mc): #vc = self passed on as parameter by controller
        self.mc = mc

    def resetDataFrame(self):
        self.df = self.original_df.copy()

    def convertFileToDataFrame(self, file): #new file opened
        self.df = pd.read_csv((file),low_memory=False,na_values="?")
        self.original_df = self.df.copy()


    def getAllColumnNames(self):
        return self.df.columns.tolist()

    def getAllNumericColumnNames(self):
        allNumericColumnNames = []
        allColNames = self.df.columns.tolist()
        for colName in allColNames:
            if self.df[colName].dtype != "object":
                allNumericColumnNames.append(colName)
        return allNumericColumnNames


    def exportCSV(self):
        #self.df = pd.read_csv((self.fileNameAndPath),low_memory=False,na_values="?")
        fileName = (os.path.basename(self.fileNameAndPath)) #obtaining actual file's name
        filePath = self.fileNameAndPath.strip(fileName) #getting rid of file name from the path
        fileName = fileName.strip(".csv") #Getting rid of .csv extension to add "_Encoded"
        fileName = fileName+"_Encoded.csv" #creating new file name
        saveDirPathWithFileName = filePath+fileName #modifying previous path with new file name
        self.cleanData()
        self.df.to_csv(saveDirPathWithFileName)

    def executeOperation(self):
        #print(self.df.columns.get_values().tolist())

        #self.df = pd.read_csv((self.fileNameAndPath),low_memory=False,na_values="?")
        self.cleanData()
        print("Features used for training: ",self.getAllColumnNames())
        self.featureColumnList = self.df.columns.tolist()
        self.featureColumnList.remove(self.className)
        if self.alternateEveryFeature:
            self.runSelectedAlgorithmOnEveryFeatureCombination(self.featureColumnList)
        elif self.findBestHyperParameters:
            self.bestHyperParameterSelector(self.featureColumnList)
        else:
            self.runSelectedAlgorithm()
            if self.seeClfReport:
                self.generateClfReport()

    def bestHyperParameterSelector(self, featureColumnList):
        print()

    def featureSelector(self, optionPicked):
        self.cleanData()
        self.df = self.df.dropna()
        featureColumnList = self.df.columns.get_values().tolist()
        featureColumnList.remove(self.className)
        X = self.df[featureColumnList]
        Y = self.df[self.className]

        if optionPicked == "Recursive Feature Elimination":
            print("RECURSIVE FEATURE ELIMINATON")
            model = LogisticRegression()
            rfe = RFE(model, 5)
            fit = rfe.fit(X, Y)
            print("Number of features: %d" % fit.n_features_,"\n")
            logRegFeaturesSelected = dict(zip(featureColumnList, fit.support_))
            print("FEATURES SELECTED: ",logRegFeaturesSelected,"\n")
            logRegFeatureRanking = dict(zip(featureColumnList, fit.ranking_))
            sortedFeatureRanking = sorted(logRegFeatureRanking.items(), key=operator.itemgetter(1))
            print("FEATURE RANKING: ",sortedFeatureRanking,"\n")
            print("First %d features have been selected. Number of feature selected modifiable." %fit.n_features_)

        elif optionPicked == "Feature Importance":
            print("FEATURE IMPORTANCE")
            model = ExtraTreesClassifier()
            model.fit(X, Y)
            roundedFeatureImp = [round(elem, 3) for elem in model.feature_importances_ ]
            ExtraTreesFeatureRanking = dict(zip(featureColumnList, roundedFeatureImp))
            ExtraTreesFeatureRankingSORTED = sorted(ExtraTreesFeatureRanking.items(), key=operator.itemgetter(1))
            print("The larger score the more important the attribute\n",ExtraTreesFeatureRankingSORTED)

        elif optionPicked == "Univariate Selection":
            print("UNIVARIATE SELECTION")
            scoreFuncs = ["f_classif", "mutual_info_classif", "f_regression", "mutual_info_regression", "SelectPercentile", "SelectFpr", "SelectFdr", "SelectFwe", "GenericUnivariateSelect"]
            #for func in scoreFuncs:
            #print(func)
            test = SelectKBest(score_func=mutual_info_classif, k=3)
            fit = test.fit(X, Y)
            # summarize scores
            np.set_printoptions(precision=3)
            #print("Scoring function: ", func)
            print("Summarized scores:\n",fit.scores_)
            features = fit.transform(X)
            # summarize selected features
            print("\nSummarized selected features:\n",features[0:5,:])

        elif optionPicked == "Principal Component Analysis":
            print("PRINCIPAL COMPONENT ANALYSIS")
            pca = PCA(n_components=3)
            fit = pca.fit(X)
            # summarize components
            print("Explained Variance: %s" % fit.explained_variance_ratio_)
            print(fit.components_)

    def getDataSummary(self):
        uniqueValueOfEachColumn = {}
        for colName in self.df.columns:
            uniqueValueOfEachColumn[colName] = len(self.df[colName].unique())

        unprocessedRowCount = self.df.shape[0]
        unprocessedColumnCount = self.df.shape[1]
        temp_df = self.df.dropna()
        unprocessedNullRowCount = unprocessedRowCount - temp_df.shape[0]
        nullRowPct = round((unprocessedNullRowCount/unprocessedRowCount)*100)
        strRowAndColCount = ("Total column count: %d   Total row count: %d   Null row count: %d(%d%%)" %(unprocessedColumnCount,unprocessedRowCount,unprocessedNullRowCount,nullRowPct))

        #Storing names of columns in an array
        temp_df = self.df.columns.get_values()
        columnNameArray = temp_df.tolist()

        columnNameWithNullPercentageDictionary = {}

        #Store column names and their null percentage in a dictionary
        for i in range(unprocessedColumnCount):
            columnName = columnNameArray[i]
            nullValCount = sum(pd.isnull(self.df[columnName]))
            nullPercentage = int((nullValCount/unprocessedRowCount)*100)
            columnNameWithNullPercentageDictionary[columnName] = nullPercentage

        return uniqueValueOfEachColumn, strRowAndColCount, columnNameWithNullPercentageDictionary
 

    def generateHistogram(self,colName):
        #Getting rid of null values:
        self.df.dropna(inplace = True)
        sns.set(rc={'figure.figsize':(8,6)})
        if self.df[colName].dtype == "object": #Column contains string data
            #Bar plots help visualize the distributions of categorical variables
            # Count Plot (a.k.a. Bar Plot)
            fig = sns.countplot(x=colName, data=self.df)            
            # Rotate x-labels
            if len(self.df[colName].unique()) > 5:
                plt.xticks(rotation=90)
                plt.tight_layout()
            title = colName+" Bar Plot"
            fig = plt.gcf()
            fig.canvas.set_window_title(title) 
            plt.show()
        else: #Column contains numeric data
            #Histograms allow you to plot the distributions of numeric variables.
            fig = sns.distplot(self.df[colName])
            plt.ticklabel_format(style='plain', axis='x')
            title = colName+" Histogram"
            fig = plt.gcf()
            fig.canvas.set_window_title(title) 
            plt.show()

    def generateScatterPlot(self,x,y):
        self.df.dropna(inplace = True)
        sns.set(rc={'figure.figsize':(8,6)})
        sns.pairplot(self.df, x_vars=x, y_vars=y, size=7, aspect=1, kind='reg')
        title = y+" vs "+x+" Scatter Plot"
        fig = plt.gcf()
        fig.canvas.set_window_title(title) 
        plt.show()

    def generateClfReport(self):
        print()



    def runSelectedAlgorithmOnEveryFeatureCombination(self, featureColumnList):
        bestScore = 0
        bestFeatureSubset = []
        iterationCount = 1
        try:
            for features in range(len(featureColumnList)+1, 0, -1):
            #for features in range(1,len(featureColumnList)+1):
                for currentFeatureSubset in itertools.combinations(featureColumnList, features):
                    currentFeatureSubsetArray = np.array(currentFeatureSubset)
                    startTime = time.time()
                    self.runSelectedAlgorithm()
                    totalTimeTaken = round((time.time()-startTime),2)
                    if self.score > bestScore:
                        bestScore = self.score
                        bestFeatureSubset = currentFeatureSubset
                        self.strBestFeatures = ("Best score of: %0.2f%% obtained at iteration number: %d from feature set: %s\n"%(bestScore, iterationCount,bestFeatureSubset))
                        print(self.strBestFeatures)

                    #csvRow = np.array([iterationCount,score,totalTimeTaken,len(currentFeatureSubset),currentFeatureSubset,bestScore],dtype=object)
                    self.strCurrentFeatures = ("Iteration Count: %d Current score: %0.3f%%  Time taken: %0.2fs Current Feature Set: %s\n" % (iterationCount, self.score, totalTimeTaken, currentFeatureSubset))
                    print(self.strBestFeatures)
                    print(self.strCurrentFeatures)
                    iterationCount = iterationCount + 1

        except KeyboardInterrupt:
            print("User cancelled program at iteration number %d"%iterationCount)

        print("Best scored of %0.2f%% obtained from feature set: %s"%(bestScore,bestFeatureSubset))

    def labelEncode(self,columnName):
        self.df[columnName] = self.df[columnName].astype('category')
        labelEncodedColumnName = columnName+"_LabelEncoded"
        self.df[labelEncodedColumnName] = self.df[columnName].cat.codes
        self.df.drop([columnName], 1, inplace=True)


    def oneHotEncode(self,columnName):
        #global df
        uniqueLabels = self.df[columnName].unique().tolist() #unique labels of the column passed
        pd.get_dummies(self.df, columns=[columnName]).head()
        #Dropping original column and adding dummy columns to original dataframe
        self.df = pd.concat([self.df.drop(columnName, axis=1), pd.get_dummies(self.df[columnName])], axis=1)
        #Renaming columns
        for column in self.df:
            for label in uniqueLabels:
                if column == label:
                    newColName = columnName+"_"+column+"_OneHotEncoded"
                    self.df.rename(columns={column: newColName}, inplace=True)

    def cleanData(self):
        if self.dropColumnsManually == True:
            for col in self.userDefinedColumnsToBeDropped:
                self.df.drop([col], 1, inplace=True)
            print("\n%d user defined column(s) dropped: %s\n" % (len(self.userDefinedColumnsToBeDropped),self.userDefinedColumnsToBeDropped))
        #print(self.df.describe())
        unprocessedRowCount = self.df.shape[0]
        unprocessedColumnCount = self.df.shape[1]
        #print("UNPROCESSED Row Count: %d Column count: %d" %(unprocessedRowCount,unprocessedColumnCount))
        temp_df = self.df.dropna()
        unprocessedNullRowCount = unprocessedRowCount - temp_df.shape[0]
        unprocessedNullColumnCount = unprocessedColumnCount
        #print("NULL Row Count: %d Column count: %d\n" %(unprocessedNullRowCount,unprocessedNullColumnCount))


        #Storing names of columns in an array
        temp_df = self.df.columns.get_values()
        columnNameArray = temp_df.tolist()

        columnNameWithNullPercentageDictionary = {}

        #Store column names and their null percentage in a dictionary
        for i in range(unprocessedColumnCount):
            columnName = columnNameArray[i]
            nullValCount = sum(pd.isnull(self.df[columnName]))
            nullPercentage = int((nullValCount/unprocessedRowCount)*100)
            columnNameWithNullPercentageDictionary[columnName] = nullPercentage

        #Dropping columns with null value percentages greater than preset threshold
        droppedColumnWithNullPercentageDictionary = {} #Contains record of dropped columns
        for i in range(unprocessedColumnCount):
            columnName = columnNameArray[i]
            if columnNameWithNullPercentageDictionary[columnName] > self.nullPercentageAtWhichColumnsAreDropped:
                if columnName != self.className:
                    droppedColumnWithNullPercentageDictionary[columnName] = columnNameWithNullPercentageDictionary[columnName]
                    self.df.drop([columnName], 1, inplace=True)
        columnsDropped = unprocessedColumnCount - self.df.shape[1]
        #print("Number of columns that exceeded null percentage (%d) threshold: %d"%(self.nullPercentageAtWhichColumnsAreDropped,(unprocessedColumnCount - df.shape[1])))
        print("%d high null percentage dropped column(s): %s\n"%(columnsDropped,droppedColumnWithNullPercentageDictionary))
        labelEncodedDictionary = {}
        oneHotEncodedArray = []
        for column in self.df:
            if self.df[column].dtype == "object": #encode only if column is of type object
                labelCount = len(self.df[column].value_counts())
                if labelCount > self.maxLabelCountWhereOneHotEncodePerformed:
                    if column == self.className: #one hot encoding class will result in more than 1 class column
                        self.className = self.className+"_LabelEncoded"
                    labelEncodedDictionary[column] = labelCount
                    self.labelEncode(column)
                else:
                    if column == self.className: #one hot encoding class will result in more than 1 class column
                        labelEncodedDictionary[column] = labelCount
                        self.className = self.className+"_LabelEncoded"
                        self.labelEncode(column)
                    else:
                        oneHotEncodedArray.append(column)
                        self.oneHotEncode(column)
        #print("Number of columns that have been-\nLabel encoded: %d, One hot encoded: %d"%(labelEncodeCount,oneHotEncodeCount))

    def linearRegression(self):
        #print("Executing Linear Regression")
        lr_df = self.df
        lr_df = self.df.dropna()
        # use the list to select a subset of the DataFrame (X)
        X = lr_df[self.featureColumnList]
        # select the Sales column as the response (y)
        y = lr_df[self.className]
        lm = LinearRegression()
        if self.performCrossValidation:
            #print("With cross-validation")
            # 10-fold cross-validation
            scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
            # fix the sign of MSE scores
            mse_scores = -scores
            # convert from MSE to RMSE
            rmse_scores = np.sqrt(mse_scores)
            lr_accuracy = rmse_scores.mean()
        else:
            X_train, X_test, y_train, y_test = train_test_split(lr_df.drop(self.className, axis=1),lr_df[self.className], test_size=0.33)
            lm.fit(X_train, y_train)
            preds = lm.predict(X_test)
            lr_accuracy = lm.score(X_test, y_test)
        # calculate the average RMSE
        
        return lr_accuracy

    def runSelectedAlgorithm(self):
        if self.algorithmChosen == "Linear Regression":
            self.score = self.linearRegression()
        elif self.algorithmChosen == "Logistic Regression":
            self.score = self.getModelScore(LogisticRegression())
        elif self.algorithmChosen == "Linear Discriminant Analysis":
            self.score = self.getModelScore(LinearDiscriminantAnalysis())
        elif self.algorithmChosen == "XG Boost":
            self.score = self.getModelScore(XGBClassifier(n_estimators=100))
        elif self.algorithmChosen == "Random Forest Classifier":
            self.score = self.getModelScore(RandomForestClassifier(n_estimators=100))
        elif self.algorithmChosen == "Gaussian Naive Bayes":
            self.score = self.getModelScore(GaussianNB())
        elif self.algorithmChosen == "K Nearest Neighbor":
            self.score = self.getModelScore(KNeighborsClassifier(n_neighbors=6))
        elif self.algorithmChosen == "Decision Tree Classifier":
            self.score = self.getModelScore(tree.DecisionTreeClassifier())
        elif self.algorithmChosen == "Support Vector Machine":
            self.score = self.getModelScore(svm.SVC())
        elif self.algorithmChosen == "Gradient Boosting Machine":
            self.score = self.getModelScore(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
        elif self.algorithmChosen == "Multilayer Perceptron":
            self.score = self.getModelScore(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))


    def getModelScore(self, clf):
        startTime = time.time()        
        algo_df = self.df.copy()
        algo_df = algo_df.dropna()

        X = algo_df[self.featureColumnList]
        y = algo_df[self.className]

        if self.performCrossValidation:
            algo_accuracy = cross_val_score(clf, X, y, cv=10, scoring = "accuracy").mean()

        else:
            X_train, X_test, y_train, y_test = train_test_split(algo_df.drop(self.className, axis=1),algo_df[self.className], test_size=0.33)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            algo_accuracy = (preds == y_test).sum().astype(float) / len(preds)

        timeTaken = time.time() - startTime
        print("Time taken: %0.1fs" %timeTaken)
        return round(algo_accuracy,4)*100

    def getClfReport():
        print("clf rep")
