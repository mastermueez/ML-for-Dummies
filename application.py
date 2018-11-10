# Model: Data Structure. Controller can send messages to it, andcurrentFeatureSelectionOption model can respond to message.
# View : User interface elements. Controller can send messages to it. View can call methods from Controller when an event happens.
# Controller: Ties View and Model together. turns UI responses into chages in data.

from controller import *
from tkinter import *

def main():
    root = Tk()
    app = Controller(root)
    root.mainloop()  
 
if __name__ == '__main__':
    main()  