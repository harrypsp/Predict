# Importing Libaries for website development
from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename

# Importing the Libaries used within the Machine Learning Code
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Functions

# Read Dataset Function
# Used to read dataset path to determine if csv file then create dataframe.
def readDataset(path_to_dataset):
    path_to_dataset = str(path_to_dataset)
    if (path_to_dataset.endswith('.csv')):
        input_data = pd.read_csv(path_to_dataset)
        print("Reading...")
        return input_data
    else:
        print("File Type not Supported")

# Pre Processor for datasets
def preProcessing(dataset):
    df_one_hot_protocol_type = pd.get_dummies(dataset['protocol_type'])
    if(len(df_one_hot_protocol_type.columns) > 1):
        df_one_hot_protocol_type = pd.get_dummies(dataset['protocol_type'], drop_first=True)

    df_one_hot_service = pd.get_dummies(dataset['service'])
    if(len(df_one_hot_service.columns) > 1):
        df_one_hot_service = pd.get_dummies(dataset['service'], drop_first=True)

    df_one_hot_flag = pd.get_dummies(dataset['flag'])
    if(len(df_one_hot_flag.columns) > 1):
        df_one_hot_flag = pd.get_dummies(dataset['flag'], drop_first=True)

    df_duration = dataset.iloc[:, 0]

    dataset = dataset.iloc[:, 4:42]
    dataset = pd.concat([df_one_hot_flag, dataset], axis=1) # Adds flag

    dataset = pd.concat([df_one_hot_service, dataset], axis=1) # Adds service

    dataset = pd.concat([df_one_hot_protocol_type, dataset], axis=1) # Adds protocol

    dataset = pd.concat([df_duration, dataset], axis=1) # Adds duration


    return dataset

# Trainning function which fits classifiers
def training(training_Dataset, learning_Method, feature_Amount):

    X = training_Dataset.iloc[:, :-1]
    y = training_Dataset.iloc[:, 115]

    # Encoding Outcome
    y = LabelEncoder().fit_transform(y)

    # Feature Selection

    bestfeatures = SelectKBest(score_func=chi2, k=feature_Amount)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(training_Dataset.columns)

    # New features
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    new_features = featureScores.nlargest(feature_Amount,'Score')
    new_features = new_features.drop("Score", axis=1)

    # New features list created
    new_features_for_training = new_features.iloc[:, :1]
    new_features_for_training = new_features_for_training.index.tolist()

    X = training_Dataset.iloc[:, new_features_for_training].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # Fitting the Classifier to the training set
    if learning_Method == "Support Vector Machine":
        classifier = SVC(kernel='rbf')
        print("SVM Selected")
    elif learning_Method == "Naive Bayes":
        classifier = GaussianNB()
        print("Naive Bayes Selected")
    else:
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        print("KNN Selected")

    # Fitting Classifier
    print("Fitting Classifier...")
    startTimer = datetime.now()
    classifier.fit(X_train, y_train)
    print("Classifier Fitted")
    print("Time taken:",datetime.now() - startTimer)
    startTimer = datetime.now()
    print("Training...")
    y_pred = classifier.predict(X_test)
    print("Time taken:",datetime.now() - startTimer)

    classifier_Accuracy = accuracy_score(y_test, y_pred)


    # Removing prediction related variables

    return classifier, classifier_Accuracy, new_features_for_training, new_features

# Prediction function to make predictions on new data
def prediction(classifier, processed_Input, new_features_for_training, column_names, label_names):

    processed_Input_Final = column_names.iloc[:, :]
    processed_Input_Final = pd.concat([processed_Input_Final, processed_Input], axis=0, ignore_index=True, sort=False)
    processed_Input_Final = processed_Input_Final.iloc[:, new_features_for_training]

    processed_Input_Final.fillna(0, inplace=True)
    X_prediction = processed_Input_Final.iloc[:, :].values

    sc_X = StandardScaler()
    X_prediction = sc_X.fit_transform(X_prediction)

    print("Making Prediction...")
    startTimer = datetime.now()
    result = classifier.predict(X_prediction)
    print("Prediction Made")
    print("Time taken:",datetime.now() - startTimer)

    naming_scheme = pd.DataFrame(data=None, columns=label_names.columns)
    result_decode = pd.DataFrame(result)
    result_decode = pd.get_dummies(result_decode[0])
    columns = list(result_decode.columns.values)
    naming_scheme = naming_scheme.iloc[:, columns]

    naming_scheme = list(naming_scheme.columns)
    result_decode.columns = naming_scheme
    result_decode = result_decode.idxmax(axis=1)
    prediction_Result = result_decode.value_counts()

    packets_Return = processed_Input_Final
    packets_Return['label'] = result_decode

    return prediction_Result, packets_Return

# Main Function which operates the program
def main(input_File, training_Input, learning_Method, feature_Amount):
# Dataset Import

    training_Dataset = readDataset(training_Input)
    print("Training Set Found !")
    training_Dataset.drop_duplicates(subset=None, keep='first', inplace=True)

    input_Dataset = readDataset(input_File)
    print("Input File Found !")
    input_Dataset.drop_duplicates(subset=None, keep='first', inplace=True)
    input_Types = input_Dataset['label'].value_counts()


    processed_Dataset = preProcessing(training_Dataset)
    print("Processing Training Data...")

    processed_Input = preProcessing(input_Dataset)
    print("Processing Input Data...")


    column_names = pd.DataFrame(data=None, columns=processed_Dataset.columns)
    label_names = pd.get_dummies(processed_Dataset['label'])

    classifier, classifier_Accuracy, new_features_for_training, new_features = training(processed_Dataset, learning_Method, feature_Amount)
    print("Classifier's Estimated Accuracy:")
    print(classifier_Accuracy)

    prediction_Result, packets_Return = prediction(classifier, processed_Input, new_features_for_training, column_names, label_names)


    return prediction_Result, input_Types, new_features, packets_Return, classifier_Accuracy

# Defining flask app
app = Flask(__name__)

# Defining where files will be uploaded
app.config['UPLOADED_FILES_DEST'] = 'C:/Users/Harry_McElwee/Documents/Diss/Predict/static/datasets'

# Creating a home page
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', title='Home')
    # render_template will link to the templates folder

# Creating a predict page
@app.route('/predict', methods=["GET", "POST"])
def predict():
    return render_template('predict.html', title='Predict')


@app.route('/results', methods=["GET", "POST"])
def results():
    if request.method == 'POST':
        if request.files:
            training_file = request.files["training_dataset"]
            train_filename = secure_filename(training_file.filename)
            training_file.save(os.path.join(app.config["UPLOADED_FILES_DEST"], train_filename))
            file = request.files["dataset"]
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOADED_FILES_DEST"], filename))

            learning_Method = request.form['method_Select']
            feature_Amount = request.form['feature_Amount']

            feature_Amount = int(feature_Amount)
            # model_select =
            # Machine Learning declaration
            training_Input = os.path.join('static/datasets', train_filename)
            input_File = os.path.join('static/datasets', filename)

            try:
                prediction_Result, input_Types, new_features, packets_Return, classifier_Accuracy = main(input_File, training_Input, learning_Method, feature_Amount)

                classifier_Accuracy = classifier_Accuracy*100
                classifier_Accuracy = np.round(classifier_Accuracy, decimals=2)

                overall_Result = input_Types
                overall_Result = pd.concat([overall_Result, prediction_Result], axis=1, sort=True)
                overall_Result.columns = ['Actual Dataset', 'Prediction Result']
                overall_Result['Accuracy (%)'] = np.divide(overall_Result['Prediction Result'], overall_Result['Actual Dataset'], where=overall_Result['Actual Dataset'] != 0)
                overall_Result['Accuracy (%)'] = np.multiply(overall_Result['Accuracy (%)'], 100)
                overall_Result['Accuracy (%)'] = np.round(overall_Result['Accuracy (%)'], decimals=2)

                bad_Packets=packets_Return[~packets_Return['label'].isin(['normal'])]
                bad_Packets_Result = bad_Packets['label']
                bad_Packets = bad_Packets.index.tolist()
                bad_Packets_Dataset = pd.read_csv(input_File)
                bad_Packets_Dataset = bad_Packets_Dataset.loc[bad_Packets, :]
                bad_Packets_Dataset = bad_Packets_Dataset.drop('label', axis=1)
                bad_Packets_Dataset['label'] = bad_Packets_Result
                packets_Return = bad_Packets_Dataset
                del bad_Packets
                del bad_Packets_Dataset
                del bad_Packets_Result

                return render_template('results.html', title='Results', Result=overall_Result, features_Used = new_features.reset_index(drop=True), packets_Analysis=packets_Return, model=learning_Method, model_Score=classifier_Accuracy)
            except:
                error = "Catch all error handle. Something terrible has gone wrong"
                return render_template('home.html', title='Error', Error=error)
    else:
        return render_template('home.html', title='Error')

if __name__ == '__main__':
    app.run(debug=True)
