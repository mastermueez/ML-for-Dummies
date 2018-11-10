#
#Controller: Ties View and Model together.
#       --Performs actions based on View events.
#       --Sends messages to Model and View and gets responses
#       --Has Delegates 

from model import *
from view import *
#from tkinter import *


class Controller():
    def __init__(self,root): #parent is root from application
        self.root = root
        self.model = Model(self)    # initializes the model
        self.view = View(self,root)  #initializes the view     

    def initializeModel(self):
        entries = self.view.getAllEntries()
        self.model.className = entries[0]
        self.model.nullPercentageAtWhichColumnsAreDropped = int(entries[1])
        self.model.maxLabelCountWhereOneHotEncodePerformed = int(entries[2])
        self.model.algorithmChosen = self.view.getAlgorithmChosen()
        self.model.performCrossValidation = self.view.getCrossValidationSelection()
        self.model.seeClfReport = self.view.getClfReportSelection()
        self.model.dropColumnsManually = self.view.getDropColumnsManuallySelection()
        if self.model.dropColumnsManually: #If user has manually selected columns to be dropped
            self.model.userDefinedColumnsToBeDropped = self.view.getCheckedList(self.model.getAllColumnNames())
        self.model.alternateEveryFeature = self.view.getAlternateEveryFeatureSelection() #If user wants to find the best features
        #self.model.findBestHyperParameters = self.view.getFindBestHyperParameterSelection()


    #EVENT HANDLERS
    def generateCSVButtonPressed(self):
        try:
            self.initializeModel()
            self.model.exportCSV()
            self.model.resetDataFrame()
            self.view.displayInfoMessage("File exported successfully")
        except Exception as exc:
            self.model.resetDataFrame()
            self.view.displayWarningMessage(exc)

    def runButtonPressed(self):
        #try:
        self.initializeModel()
        self.model.executeOperation()

        if self.model.seeClfReport:
            windowName = self.model.algorithmChosen+" Classification Report"
            self.view.displayInfoInNewWindow(windowName,self.model.clfReportInfo,False)
        else:
            self.view.displayResult(self.model.algorithmChosen, self.model.score, self.model.performCrossValidation)
        self.model.resetDataFrame()
        #except Exception as exc:
        #    self.model.resetDataFrame()
        #    self.view.displayErrorMessage(exc)

    def openFileButtonPressed(self, file):
        self.model.fileNameAndPath = file #every time a new file is uploaded, model will be updated
        try:
            self.model.convertFileToDataFrame(file)
            allColumnNames = self.model.getAllColumnNames()
            className = self.view.getEntryBasedOnIndex(0)
            classNameExists = className in allColumnNames
            #File has been opened
            self.view.createDataSummaryBtn()

            if classNameExists:
                allColumnNames.remove(className)
            else:
                errorMsg = "'"+className+"'"+" is not a feature of the chosen file. Please edit the target class field correctly and REOPEN the file."
                self.view.displayErrorMessage(errorMsg)

            dropColumnsManually = self.view.getDropColumnsManuallySelection()
            if dropColumnsManually and classNameExists:
                self.view.createChecklist(allColumnNames)
        except:
            try:
                self.model.resetDataFrame()
            except:
                self.view.displayWarningMessage("You did not select any file")

    def dataSummaryBtnPressed(self):
        allColumnNames = self.model.getAllColumnNames()
        allNumericColumnNames = self.model.getAllNumericColumnNames()
        uniqueVal, strRowAndColCount, columnNameWithNullPercentageDictionary = self.model.getDataSummary()
        self.view.createDataSummaryWindow(allColumnNames, allNumericColumnNames, uniqueVal, strRowAndColCount, columnNameWithNullPercentageDictionary)

    def bestFeatureSelectorBtnPressed(self):
#        try:
        #entries = self.view.getAllEntries()
        #self.model.className = entries[0]
        #self.model.nullPercentageAtWhichColumnsAreDropped = int(entries[1])
        #self.model.maxLabelCountWhereOneHotEncodePerformed = int(entries[2])
        self.initializeModel()
        self.model.featureSelector(self.view.getFeatureSelectorSelection())

        self.view.displayInfoInNewWindow(self.model.clfReportInfo.pop(0),self.model.clfReportInfo,True)
        self.model.resetDataFrame()
    #    except: 
     #       self.view.displayWarningMessage("You did not select any file")

    def generateHistogramBtnPressed(self):
        try:
            self.model.generateHistogram(self.view.getHistFeatureColSelection())
            self.model.resetDataFrame() #null rows had to be dropped to generate hist
        except Exception as exc:
            self.model.resetDataFrame()
            self.view.displayErrorMessage(exc)


    def generateScatterPlotBtnPressed(self):
        x = self.view.getScatterPlotXSelection()
        y = self.view.getScatterPlotYSelection()
        self.model.generateScatterPlot(x,y)
        self.model.resetDataFrame() #null rows had to be dropped to generate scatter plot