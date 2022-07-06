import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import time
import streamlit as st
import graphviz as graphviz


def read_data(filename):

    data = pd.read_csv('iris_data.csv')
    return data

def split_data(data, test_size = 0.4):
    train, test = train_test_split(data, test_size = test_size)

    return train, test

def plot_data_distribution(data):
    st.write("Plotting the Distribution of length and width")
    n_bins = 10
    fig, axs = plt.subplots(2, 2)

    # Sepal Length
    axs[0,0].hist(data['sepal_length'], bins = n_bins)
    axs[0,0].set_title('Sepal Length')

    # Sepal Width
    axs[0,1].hist(data['sepal_width'], bins = n_bins)
    axs[0,1].set_title('Sepal Width')

    # Petal Length
    axs[1,0].hist(data['petal_length'], bins = n_bins)
    axs[1,0].set_title('Petal Length')

    # Petal Width
    axs[1,1].hist(data['petal_width'], bins = n_bins)
    axs[1,1].set_title('Petal Width')
    # add some spacing between subplots
    fig.tight_layout(pad=1.0)

    return fig

# STREAMLIT FUNCTIONS
def fn_upload_file():

    uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)

    #Checking if file is uploaded or not
    if uploaded_file is not None:

        #Checking if the uploaded file is of correct type
        if(uploaded_file.type == 'text/csv'):
            dataframe = pd.read_csv(uploaded_file)
            return dataframe

        else:
            st.error("Please upload a file of the correct type '(.csv)'")

    else:
        st.warning("File not uploaded")

def progress_bar_example():
    st.write("Loading...")
    my_bar = st.progress(0)
    num = 50

    for i in range(num):
        time.sleep(0.1)
        my_bar.progress( (i+1)/num )


def balloons_example():
    st.write("Loading ... ")
    st.write("Loading Completed ")
    st.balloons()

if __name__ == '__main__':

    st.write("Making a classification model for the IRIS dataset.")

    #filename = 'iris_data.csv'
    #data = read_data(filename)

    data = fn_upload_file()

    if(data is not None):
        train, test = split_data(data)
        
        analysis_checkbox_answer = st.checkbox('Show Analysis')
        #train_checkbox_answer = st.checkbox('Train Model')
        
        features = ['sepal_length','sepal_width','petal_length','petal_width']
        classes = list(np.unique(data['species']))
        X_train = train[features]
        y_train = train['species']
        X_test = test[features]
        y_test = test['species']
        
        if(analysis_checkbox_answer == True):
            fig = plot_data_distribution(train)
            st.write(data)
            st.write(fig)

        results_checkbox_answer = st.checkbox('Show Results')

        if(results_checkbox_answer == True):

            #spinner_example()
            #Train the model
            model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
            model = model.fit(X_train,y_train)
            
            progress_bar_example()
            # Predict and calculate accuracy
            prediction = model.predict(X_test)
            accuracy = metrics.accuracy_score(prediction, y_test)

            visualize_results_checkbox_ans = st.checkbox("Visualize the Results")

            if(visualize_results_checkbox_ans == True):
                st.write('The accuracy of the Decision Tree is',"{:.3f}".format(accuracy))
                dot_data  = tree.export_graphviz(model, out_file=None, feature_names=features, class_names=classes)
                st.graphviz_chart(dot_data)
            
                balloons_example()