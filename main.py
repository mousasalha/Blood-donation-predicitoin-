import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import Frame

#for Descision tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#for naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

class BloodDonationApp:
    def __init__(self ,root):

        self.df = pd.read_csv("dataset.csv")
        '''-----------------------------------GUI---------------------------------'''
        self.root = root

        
        self.root.title("Blood Donation Predictor")

        self.root.geometry("1400x750")

            
        self.root.config(bg="#7D0A0A")

        
        

        # Left down Frame
        self.left_frame = Frame(root, bd=2, relief="groove", bg="#DFD3C3")  
        self.left_frame.grid(row=15, column=0, rowspan=11, padx=10, pady=10, sticky="nsew")


        # left Frame
        self.l_frame = Frame(root, bd=2, relief="groove", bg="#DFD3C3" , height=10 , width=40)  
        self.l_frame.grid(row=0, column=0, rowspan=11, padx=10, pady=10, sticky="nsew" )

        # right Frame
        self.right_frame = Frame(root, bd=2, relief="groove", bg="#DFD3C3")  
        self.right_frame.grid(row=2, column=2, rowspan=11, padx=10, pady=10, sticky="nsew")

        # right down Frame
        self.r_frame = Frame(root, bd=2, relief="groove", bg="#DFD3C3", height=10 , width=40)  
        self.r_frame.grid(row=15, column=2, rowspan=11, padx=10, pady=10, sticky="nsew")


        self.label_testSize = tk.Label(self.l_frame, text="Test Size (0.xx):")
        self.label_Receny = tk.Label(self.left_frame, text="Recency (months) :")
        self.label_Frequency = tk.Label(self.left_frame, text="Frequency (times) :")
        self.label_Monetary= tk.Label(self.left_frame, text="Monetary (c.c. blood) :")
        self.label_Time= tk.Label(self.left_frame, text="Time (months) :")

        self.label_testSize.grid(row=3, column=0, padx=10, pady=20)
        self.label_Receny.grid(row=9, column=0, padx=10, pady=20)
        self.label_Frequency.grid(row=10, column=0, padx=10, pady=20)
        self.label_Monetary.grid(row=11, column=0, padx=10, pady=20)
        self.label_Time.grid(row=12, column=0, padx=10, pady=20)

        self.entry_testSize= tk.Entry(self.l_frame)  
        self.entry_Receny= tk.Entry(self.left_frame)
        self.entry_Frequency  = tk.Entry(self.left_frame)
        self.entry_Monetary= tk.Entry(self.left_frame)
        self.entry_Time = tk.Entry(self.left_frame)

        self.entry_testSize.grid(row=3, column=1, padx=10, pady=20)
        self.entry_Receny.grid(row=9, column=1, padx=10, pady=20)
        self.entry_Frequency.grid(row=10, column=1, padx=10, pady=20)
        self.entry_Monetary.grid(row=11, column=1, padx=10, pady=20)
        self.entry_Time.grid(row=12, column=1, padx=10, pady=20)


        self.button_trainD = tk.Button(self.l_frame, text="Print Report for Descision Tree" , command=self.train_model_D)
        self.button_trainN = tk.Button(self.l_frame, text="Print Report for Naive Bayes", command=self.train_model_N)
        
        
        self.button_trainD.grid(row=4,column=0,padx=10 ,pady=20)
        self.button_trainN.grid(row=5,column=0,padx=10 ,pady=20)
        

        self.button_testbyD = tk.Button(self.left_frame, text="Test By Descision Tree" , command=self.test_new_data_D)
        self.button_testbyN = tk.Button(self.left_frame, text="Test By Naive Bayes", command=self.test_new_data_N)

        self.button_testbyD.grid(row=13,column=0,padx=10 ,pady=10)
        self.button_testbyN.grid(row=13,column=1,padx=10 ,pady=10)


        # Add a Text widget for output
        self.output_text = tk.Text(self.right_frame, height=20, width=60)
        self.output_text.grid(row=0, column=0, padx=10, pady=10)

        # Add a Text widget for output
        self.output_text2 = tk.Text(self.right_frame, height=20, width=60)
        self.output_text2.grid(row=0, column=1, padx=10, pady=10)
        
        # Add a Text widget for output
        self.output_text1 = tk.Text(self.r_frame, height=20, width=80)
        self.output_text1.grid(row=0, column=0, padx=10, pady=10)

        
        
        
    '''-----------------------------------Descision Tree---------------------------------'''  

    def train_model_D(self ):
        testsize = self.entry_testSize.get()
        testsize = float(testsize)
        
        self.X = self.df.drop('whether he/she donated blood in March 2024', axis=1)
        self.y = self.df['whether he/she donated blood in March 2024']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=testsize, shuffle=False)
        self.dt_classifier = DecisionTreeClassifier()
        self.dt_classifier.fit(self.X_train, self.y_train)

        y_pred = self.dt_classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        classification_report_text = classification_report(self.y_test, y_pred)
        
        
        self.output_text.delete("1.0", "end")
        # Insert the results into the Text widget
        result_text = "Descision Tree","\n".join([
            f'\nAccuracy: {accuracy:.2f}',
            'Classification Report:\n' + classification_report_text,
            'Confusion Matrix:\n' + str(confusion_matrix(self.y_test, y_pred)),
        ])
        self.output_text.insert("1.0", result_text)

    def test_new_data_D(self):
        receny = self.entry_Receny.get()
        frequency = self.entry_Frequency.get()
        monetary = self.entry_Monetary.get()
        time = self.entry_Time.get()

        receny = int(receny)
        frequency = int(frequency)
        monetary = int(monetary)
        time = int(time)

        user_data = [[receny, frequency, monetary, time]]
        prediction = self.dt_classifier.predict(user_data)

        if prediction[0] == 1:
            result = "The user is likely to donate blood as test by Descision Tree."
        else:
            result = "The user is less likely to donate bloodas test by Descision Tree."

        self.output_text1.insert("1.0", "\nPrediction Result for Descision Tree :\n" + result)

    '''-----------------------------------Niave Bayes ---------------------------------'''   

    def train_model_N(self):
        testsize = self.entry_testSize.get()
        testsize = float(testsize)

        

        X = self.df.drop('whether he/she donated blood in March 2024', axis=1)
        y = self.df['whether he/she donated blood in March 2024']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, shuffle=False)

        self.model = GaussianNB()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        
        Classification_Report = classification_report(y_test, y_pred)

        self.output_text2.delete("1.0", "end")
        # Insert the results into the Text widget
        result_text = "Naive Bayes","\n".join([
            f'Accuracy: {accuracy:.2f}',
            'Classification Report:\n' + Classification_Report,
            'Confusion Matrix:\n' + str(confusion_matrix(y_test, y_pred)),
        ])
        self.output_text2.insert("1.0", result_text)
    

    def test_new_data_N(self):
        receny = self.entry_Receny.get()
        frequency = self.entry_Frequency.get()
        monetary = self.entry_Monetary.get()
        time = self.entry_Time.get()

        receny = int(receny)
        frequency = int(frequency)
        monetary = int(monetary)
        time = int(time)

        user_input = [[receny, frequency, monetary, time]]
        prediction = self.model.predict(user_input)

        if prediction[0] == 1:
            result = "The user is likely to donate blood as test by Naive Bayes."
        else:
            result = "The user is less likely to donate blood as test by Naive Bayes."

        self.output_text1.insert("1.0", "\nPrediction Result for Naive Bayes :\n" + result)






if __name__ == "__main__":
    root = tk.Tk()
    app = BloodDonationApp(root)
    root.mainloop()        