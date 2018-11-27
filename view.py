from tkinter import *
from tkinter import filedialog, messagebox

#View : User interface elements.
#       --Controller can send messages to it.
#       --View can call methods from Controller vc when an event happens.
#       --NEVER communicates with Model.
#       --Has setters and getters to communicate with controller

class View():
    defaultTargetClass = "Success"
    #UI ELEMENTS
    fontType = "COURIER"
    fontSize = 18

    windowBGColor = "WHITE"
    labelBGColor = windowBGColor
    labelFGColor = "BLACK"

    entryBGColor = windowBGColor
    entryFGColor = "#1157c6"

    rowBGColor = windowBGColor

    quitBtnColor = "#ff0000"
    runBtnColor = "#24bf6a"

    paddingX = 10
    paddingY = 7

    labelTextList = ['Target class', 'Null % at which columns are dropped', 'Max label count for One Hot Encoding']
    entryTextList = []

    currentAlgorithmOption = None
    currentCVOption = None
    currentClfReportOption = None
    currentAlternateEveryFeatureOption = None
    currentHyperParameterOption = None
    currentDropColumnManuallyOption = None
    checkedList = None
    fileNameAndPath = None
    currentFeatureSelectorOption = None
    
    #NEW WINDOW VARIABLES
    currentBarChartFeatureColOption = None
    currentHistFeatureColOption = None
    currentScatterPlotXOption = None
    currentScatterPlotYOption = None

    def __init__(self,vc,root):
        self.vc = vc
        self.root = root
        self.root.title('ML for Dummies')
        self.frame = Frame(self.root)
        self.loadView()


    def loadView(self):
        self.createForm()

        #ALGORITHM OPTION MENU
        algoSelectionLabelName = "Select algorithm"
        algoDefaultOption = "Gaussian Naive Bayes"
        algoOptions = [algoDefaultOption, "Linear Regression", "Logistic Regression", "Linear Discriminant Analysis", "Decision Tree Classifier", "Random Forest Classifier","Gradient Boosting Machine","XG Boost","K Nearest Neighbor","Support Vector Machine","Multilayer Perceptron"]
        self.currentAlgorithmOption = self.createOptionMenu(algoSelectionLabelName, algoDefaultOption, algoOptions)

        #CLASSIFICATION REPORT OPTION MENU
        clfReportcurveLabelName = "See classification report"
        clfReportDefaultOption = "Yes"
        clfReportOptions = [clfReportDefaultOption, "No"]
        self.currentClfReportOption = self.createOptionMenu(clfReportcurveLabelName, clfReportDefaultOption, clfReportOptions)

        #CROSS VALIDATION OPTION MENU
        CVlabelName = "Perform cross validation"
        CVdefaultOption = "Yes"
        CVoptions = [CVdefaultOption, "No"]
        self.currentCVOption = self.createOptionMenu(CVlabelName, CVdefaultOption, CVoptions)

        #TRY EVERY FEATURE COMBINATION MENU
        featureSelectionLabelName = "Try every possible feature combination"
        featureSelectionDefaultOption = "No"
        featureSelectionOptions = [featureSelectionDefaultOption, "Yes"]
        self.currentAlternateEveryFeatureOption = self.createOptionMenu(featureSelectionLabelName, featureSelectionDefaultOption, featureSelectionOptions)

        #FIND BEST HYPER PARAMETER MENU
        """
        hyperParameterSelectionLabelName = "Find best hyperparameters"
        hyperParameterSelectionDefaultOption = "No"
        hyperParameterSelectionOptions = [hyperParameterSelectionDefaultOption, "Yes"]
        self.currentHyperParameterOption = self.createOptionMenu(hyperParameterSelectionLabelName, hyperParameterSelectionDefaultOption, hyperParameterSelectionOptions)
        """

        #DROP COLUMNS MANUALLY OPTION MENU
        dropColumnManuallyLabelName = "Drop columns manually"
        dropColumnManuallyDefaultOption = "Yes"
        dropColumnManuallyOptions = [dropColumnManuallyDefaultOption, "No"]
        self.currentDropColumnManuallyOption = self.createOptionMenu(dropColumnManuallyLabelName, dropColumnManuallyDefaultOption, dropColumnManuallyOptions)

        #FEATURE SELECTION MENU AND BUTTON
        featureSelectionLabelName = "Generate best features using"
        featureSelectionDefaultOption = "Recursive Feature Elimination"
        featureSelectionOptions = [featureSelectionDefaultOption, "Feature Importance", "Univariate Selection", "Principal Component Analysis"]
        eventHandler = self.vc.bestFeatureSelectorBtnPressed
        self.currentFeatureSelectorOption = self.createOptionMenuWithButton("Feature selector ", featureSelectionDefaultOption, featureSelectionOptions, "Generate",eventHandler, False, None)

        self.createButtons()

    def createForm(self):
        #LABELS AND ENTRIES
        index = 0
        defaultEntry = [self.defaultTargetClass, '67', '2']
        for labelText in self.labelTextList:
            row = Frame()    
            label = Label(row, width=36, text = labelText, anchor='w', font=(self.fontType, self.fontSize), bg = self.labelBGColor, fg = self.labelFGColor)

            if index != 0: # Numeric entry when index != 0
                vcmd = (self.root.register(self.validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
                entry = Entry(row, validate = 'key', validatecommand = vcmd,font=(self.fontType, self.fontSize), bg=self.entryBGColor, fg=self.entryFGColor)
            else: #String entry
                entry = Entry(row, font=(self.fontType, self.fontSize), bg=self.entryBGColor, fg=self.entryFGColor)

            entry.insert(END, defaultEntry[index])
            row.configure(bg = self.rowBGColor)
            row.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
            label.pack(side=LEFT)
            entry.pack(side=RIGHT, fill=X)
            self.entryTextList.insert(index, entry)
            index += 1

    def validate(self, action, index, value_if_allowed,
                   prior_value, text, validation_type, trigger_type, widget_name):
        if text in '0123456789':
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

    def createOptionMenu(self,labelName,defaultOption,options):
        row = Frame()
        lab = Label(row, width=37, text=labelName, anchor='w', font=(self.fontType, self.fontSize), bg=self.labelBGColor, fg = self.labelFGColor)
        row.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
        row.configure(bg=self.rowBGColor)
        lab.pack(side=LEFT)
        currentOption = StringVar()
        currentOption.set(defaultOption) # initial value

        option_Menu = OptionMenu(row, currentOption, *options)
        option_Menu.config(font=(self.fontType, self.fontSize),bg = self.entryBGColor, fg=self.entryFGColor)
        option_Menu.pack(side = RIGHT)
        return currentOption

    def createOptionMenuWithButton(self,lblName,defaultOption,options,btnName, eventHandler, displayInNewWindow, newWindow):
        if displayInNewWindow:
            row = Frame(newWindow)
        else:
            row = Frame()
        lblWidth = len(lblName)
        lbl = Label(row, width=lblWidth, text=lblName, anchor='w', font=(self.fontType, self.fontSize), bg=self.labelBGColor, fg = self.labelFGColor)

        currentOption = StringVar()
        currentOption.set(defaultOption) # initial value
        option_Menu = OptionMenu(row, currentOption, *options)

        btn = Button(row, text=btnName, command=eventHandler, font=(self.fontType, self.fontSize), fg = self.entryFGColor)

        row.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
        row.configure(bg=self.rowBGColor)

        lbl.pack(side=LEFT)
        option_Menu.config(font=(self.fontType, self.fontSize),bg = self.entryBGColor, fg=self.entryFGColor)
        option_Menu.pack(side = LEFT)

        btn.pack(side=LEFT, padx=self.paddingX, pady=self.paddingY)
        return currentOption



    def createChecklist(self,allColumnNames):
        dropTickLabelRow = Frame()
        dropTickLabel = Label(dropTickLabelRow, width=23, text="Tick columns to drop", anchor='w', font=(self.fontType, self.fontSize), bg=self.labelBGColor, fg = self.labelFGColor)
        dropTickLabelRow.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
        dropTickLabel.pack()
        maxColNameLength = max(len(key) for key in allColumnNames)
        checkListCount = 0 
        checkListRow = Frame()
        self.checkedList = []
        for col in allColumnNames:
            checkStatus = IntVar()
            if checkListCount > 3:
                checkListRow = Frame()
                checkListCount = 0
            chkBtn = Checkbutton(checkListRow, width=maxColNameLength+2, text=col, anchor='w', variable=checkStatus, font=(self.fontType, self.fontSize), bg = self.labelBGColor, fg = self.labelFGColor)
            checkListRow.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
            chkBtn.pack(side=LEFT)
            self.checkedList.append(checkStatus)
            checkListCount += 1

    def getCheckedList(self, allColumnNames):

        className = self.getEntryBasedOnIndex(0)
        print("className: ",className)
        print("allColumnNames: ",allColumnNames)
        allColumnNames.remove(className)
        userDefinedColumnsToBeDropped = []

        index = 0
        for checkSts in self.checkedList:
            if checkSts.get() == 1:
                userDefinedColumnsToBeDropped.append(allColumnNames[index])
            index += 1
        """
        print("allColumnNames: ", allColumnNames)
        for checkSts in self.checkedList:
            print(checkSts.get())
        print("userDefinedColumnsToBeDropped: ",userDefinedColumnsToBeDropped)
        """
        #self.checkedList = []
        return userDefinedColumnsToBeDropped

    def storeFileDir(self):
        self.fileNameAndPath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.vc.openFileButtonPressed(self.fileNameAndPath)

    def createButtons(self):
        btnRow = Frame()
        openFileBtn = Button(btnRow, text="Open File", command=self.storeFileDir, font=(self.fontType, self.fontSize), fg = self.entryFGColor)
        runBtn = Button(btnRow,text = 'Run', command= self.vc.runButtonPressed, font=(self.fontType, self.fontSize), fg=self.entryFGColor)
        quitBtn = Button(btnRow,text = 'Export CSV', command= self.vc.generateCSVButtonPressed, font=(self.fontType, self.fontSize), fg=self.quitBtnColor)
        
        btnRow.pack(side=BOTTOM, fill=X, padx=self.paddingX, pady=self.paddingY)
        openFileBtn.pack(side=LEFT, padx=self.paddingX, pady=self.paddingY)
        quitBtn.pack(side=RIGHT, padx=self.paddingX, pady=self.paddingY)
        runBtn.pack(side=LEFT, padx=self.paddingX, pady=self.paddingY)

    def createDataSummaryBtn(self):
        dataSummaryBtnRow = Frame()
        dataSummaryBtn = Button(dataSummaryBtnRow, text="Data Summary", command=self.vc.dataSummaryBtnPressed, font=(self.fontType, self.fontSize), fg = self.entryFGColor)
        dataSummaryBtnRow.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
        dataSummaryBtn.pack(side=LEFT, fill="none", expand=True, padx=self.paddingX, pady=self.paddingY)


    def displayResult(self, algorithmChosen, accuracyScore, performCrossValidation):
        accuracyScore = round(accuracyScore, 4)
        if performCrossValidation:
            if algorithmChosen == "Linear Regression":
                strDisplay = algorithmChosen+" RMSE Error w/ CV: "+str(accuracyScore)
            else:
                strDisplay = algorithmChosen+" accuracy w/ CV: "+str(accuracyScore)+"%"
        else:
            if algorithmChosen == "Linear Regression":
                strDisplay = algorithmChosen+" coeff of determination: "+str(accuracyScore)
            else:
                strDisplay = algorithmChosen+" accuracy: "+str(accuracyScore)+"%"
        resultRow = Frame()
        resultLabel = Label (resultRow, text=strDisplay, anchor='w', font=(self.fontType, self.fontSize), bg = self.labelBGColor, fg = self.labelFGColor)
        resultRow.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
        resultLabel.pack()


    def displayErrorMessage(self,displayText):
            messagebox.showerror("Error",displayText)

    def displayWarningMessage(self,displayText):
            messagebox.showwarning("Warning", displayText)

    def displayInfoMessage(self,displayText):
        messagebox.showinfo("Notice", displayText)

    def getAllEntries(self):
        entryStrList = []
        for entry in self.entryTextList:
            entryStrList.append(entry.get())
        return entryStrList

    def getEntryBasedOnIndex(self, index):
        return self.entryTextList[index].get()

    def getFileNameAndPath(self):
        return self.fileNameAndPath

    def getAlgorithmChosen(self):
        return self.currentAlgorithmOption.get()

    def getCrossValidationSelection(self):
        return self.convertStrVarToBool(self.currentCVOption)

    def getClfReportSelection(self):
        return self.convertStrVarToBool(self.currentClfReportOption)

    def getAlternateEveryFeatureSelection(self):
        return self.convertStrVarToBool(self.currentAlternateEveryFeatureOption)

    def getFindBestHyperParameterSelection(self):
        return self.convertStrVarToBool(self.currentHyperParameterOption)

    def getDropColumnsManuallySelection(self):
        return self.convertStrVarToBool(self.currentDropColumnManuallyOption)

    def getFeatureSelectorSelection(self):
        return self.currentFeatureSelectorOption.get()

    def convertStrVarToBool(self,strVar):
        status = False
        if strVar.get() == "Yes":
            status = True
        return status


    # GENERIC NEW WINDOW

    def displayInfoInNewWindow(self, windowName, contents, contentIsTextual):
        newWindow = Toplevel()
        newWindow.wm_title(windowName)
        for labelName in contents:
            row = Frame(newWindow)
            if contentIsTextual:
                txtWidth = 50
                txtHeight = len(labelName)/txtWidth
                txt = Text(row, height =txtHeight, width=txtWidth, font=(self.fontType, self.fontSize), bg=self.labelBGColor, fg = self.labelFGColor)
            else:
                lbl = Label(row, width=32, text=labelName, anchor='w', font=(self.fontType, self.fontSize), bg=self.labelBGColor, fg = self.labelFGColor)
            row.pack(side=TOP, fill=BOTH, padx=self.paddingX, pady=self.paddingY)
            row.configure(bg=self.rowBGColor)

            if contentIsTextual:
                txt.pack(side=LEFT, expand=True, fill = BOTH)
                txt.insert(END, labelName)
            else:
                lbl.pack(side=LEFT, expand=True, fill = BOTH)


    ################################
    # DATA SUMMARY WINDOW ELEMENTS #
    ################################


    def getBarChartFeatureColSelection(self):
        return self.currentBarChartFeatureColOption.get()

    def getHistFeatureColSelection(self):
        return self.currentHistFeatureColOption.get()

    def getScatterPlotXSelection(self):
        return self.currentScatterPlotXOption.get()

    def getScatterPlotYSelection(self):
        return self.currentScatterPlotYOption.get()

    def createDataSummaryWindow(self, allColumnNames, allNumericColumnNames, uniqueVal, strRowAndColCount, colNullPct):
        newWindow = Toplevel()
        newWindow.wm_title("Data Summary") 

        #Display row and col count info
        strRowAndColCountRow = Frame(newWindow)    
        strRowAndColCountLabel = Label(strRowAndColCountRow, width=76, text = strRowAndColCount, anchor='w', font=(self.fontType, self.fontSize), bg = self.labelBGColor, fg = self.labelFGColor)
        strRowAndColCountRow.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
        strRowAndColCountLabel.pack(side=LEFT)

        #Display null percentage of indivdial col LABEL
        maxColNullPctKey = max(len(key) for key in colNullPct) #To use as width reference
        colNullPctRow = Frame(newWindow)    
        txt = "Column names followed by their null percentage and number of unique values (null included): "
        colNullPctLabel = Label(colNullPctRow, width=len(txt), text = txt, anchor='w', font=(self.fontType, self.fontSize), bg = self.labelBGColor, fg = self.entryFGColor)
        colNullPctRow.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
        colNullPctLabel.pack(side=LEFT)

        #Display null percentage of indivdial col DATA
        numberOfColumns = 3
        numberOfColumns -= 1
        index = 0 
        colNullPctContentRow = Frame(newWindow)
        for x, y in colNullPct.items():
            strKeyAndItem = (x+": "+str(y)+"%, "+str(uniqueVal[x]))
            if index > numberOfColumns:
                colNullPctContentRow = Frame(newWindow)
                index = 0
            colNullPctContentLabel = Label(colNullPctContentRow, width=(maxColNullPctKey+10), text = strKeyAndItem, font=(self.fontType, self.fontSize), bg = self.labelBGColor, fg = self.labelFGColor)
            colNullPctContentRow.pack(side=TOP, fill=Y, padx=self.paddingX, pady=self.paddingY)
            colNullPctContentLabel.pack(side=LEFT)
            index += 1

        #Bar chart option menu
        defaultBarChartFeatureColOption = allColumnNames[0]
        eventHandler = self.vc.generateBarChartBtnPressed
        self.currentBarChartFeatureColOption = self.createOptionMenuWithButton("Bar chart axis",defaultBarChartFeatureColOption,allColumnNames,"Generate", eventHandler, True, newWindow)

        #Histogram option menu
        defaultHistFeatureColOption = allNumericColumnNames[0]
        eventHandler = self.vc.generateHistogramBtnPressed
        self.currentHistFeatureColOption = self.createOptionMenuWithButton("Histogram axis",defaultHistFeatureColOption,allNumericColumnNames,"Generate", eventHandler, True, newWindow)

        #Scatter plot option menu
        lblName = ["Scatter plot: X-axis"," Y-axis"]
        defaultScatterPlotColOption = allNumericColumnNames[0]
        eventHandler = self.vc.generateScatterPlotBtnPressed
        self.currentScatterPlotXOption, self.currentScatterPlotYOption = self.createTwoOptionMenusWithButton(lblName,defaultScatterPlotColOption,allNumericColumnNames,"Generate", eventHandler, True, newWindow)


    def createTwoOptionMenusWithButton(self,lblName,defaultOption,options,btnName, eventHandler, displayInNewWindow, newWindow):
        if displayInNewWindow:
            row = Frame(newWindow)
        else:
            row = Frame()

        #First label and option
        lblWidth = len(lblName[0])
        lbl1 = Label(row, width=lblWidth, text=lblName[0], anchor='w', font=(self.fontType, self.fontSize), bg=self.labelBGColor, fg = self.labelFGColor)
        currentOption1 = StringVar()
        currentOption1.set(defaultOption) # initial value
        option_Menu1 = OptionMenu(row, currentOption1, *options)

        #Second label and option
        lblWidth = len(lblName[1])
        lbl2 = Label(row, width=lblWidth, text=lblName[1], anchor='w', font=(self.fontType, self.fontSize), bg=self.labelBGColor, fg = self.labelFGColor)
        currentOption2 = StringVar()
        currentOption2.set(defaultOption) # initial value
        option_Menu2 = OptionMenu(row, currentOption2, *options)

        btn = Button(row, text=btnName, command=eventHandler, font=(self.fontType, self.fontSize), fg = self.entryFGColor)

        row.pack(side=TOP, fill=X, padx=self.paddingX, pady=self.paddingY)
        row.configure(bg=self.rowBGColor)

        lbl1.pack(side=LEFT)
        option_Menu1.config(font=(self.fontType, self.fontSize),bg = self.entryBGColor, fg=self.entryFGColor)
        option_Menu1.pack(side = LEFT)
        lbl2.pack(side=LEFT)
        option_Menu2.config(font=(self.fontType, self.fontSize),bg = self.entryBGColor, fg=self.entryFGColor)
        option_Menu2.pack(side = LEFT)

        btn.pack(side=LEFT, padx=self.paddingX, pady=self.paddingY)
        return currentOption1, currentOption2

