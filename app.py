from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename


# Importing the Libaries
import pandas as pd
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Functions
def readDataset(path_to_dataset):
    path_to_dataset = str(path_to_dataset)
    if (path_to_dataset.endswith('.csv')):
        input_data = pd.read_csv(path_to_dataset)
        print("Reading...")
        return input_data
    else:
        print("File type not supported")

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
    del df_one_hot_flag
    dataset = pd.concat([df_one_hot_service, dataset], axis=1) # Adds service
    del df_one_hot_service
    dataset = pd.concat([df_one_hot_protocol_type, dataset], axis=1) # Adds protocol
    del df_one_hot_protocol_type
    dataset = pd.concat([df_duration, dataset], axis=1) # Adds duration
    del df_duration

    return dataset

def training(training_Dataset):

    X = training_Dataset.iloc[:, :-1]
    y = training_Dataset.iloc[:, 115]

    # Encoding Outcome
    y = LabelEncoder().fit_transform(y)

    # Feature Selection

    bestfeatures = SelectKBest(score_func=chi2, k=20)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(training_Dataset.columns)

    # New features
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    new_features = featureScores.nlargest(20,'Score')
    new_features = new_features.drop("Score", axis=1)

    # New features list created
    new_features_for_training = new_features.iloc[:, :1]
    new_features_for_training = new_features_for_training.index.tolist()

    # Removing any now redundant variables
    del new_features
    del dfcolumns
    del dfscores
    del bestfeatures
    del featureScores
    del fit

    X = training_Dataset.iloc[:, new_features_for_training].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    del sc_X


    # Fitting the Classifier to the training set
    #classifier = GaussianNB()
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

    # Fitting Classifier
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    classifier_Accuracy = accuracy_score(y_test, y_pred)


    # Removing prediction related variables
    del X
    del X_train
    del X_test
    del y_train
    del y_test

    return classifier, classifier_Accuracy, new_features_for_training

def prediction(classifier, processed_Input, new_features_for_training, column_names, label_names):

    processed_Input_Final = column_names.iloc[:, :]
    processed_Input_Final = pd.concat([processed_Input_Final, processed_Input], axis=0, ignore_index=True, sort=False)
    processed_Input_Final = processed_Input_Final.iloc[:, new_features_for_training]

    processed_Input_Final.fillna(0, inplace=True)
    X_prediction = processed_Input_Final.iloc[:, :].values

    sc_X = StandardScaler()
    X_prediction = sc_X.fit_transform(X_prediction)

    result = classifier.predict(X_prediction)

    naming_scheme = pd.DataFrame(data=None, columns=label_names.columns)
    result_decode = pd.DataFrame(result)
    result_decode = pd.get_dummies(result_decode[0])
    columns = list(result_decode.columns.values)
    naming_scheme = naming_scheme.iloc[:, columns]

    naming_scheme = list(naming_scheme.columns)
    result_decode.columns = naming_scheme
    result_decode = result_decode.idxmax(axis=1)

    del naming_scheme
    del columns

    prediction_Result = result_decode.value_counts()
    return prediction_Result

# Main Function to be pickled
#main = main()

def main(input_File, training_Input):
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
    del training_Dataset
    processed_Input = preProcessing(input_Dataset)
    print("Processing Input Data...")
    del input_Dataset

    column_names = pd.DataFrame(data=None, columns=processed_Dataset.columns)
    label_names = pd.get_dummies(processed_Dataset['label'])

    classifier, classifier_Accuracy, new_features_for_training = training(processed_Dataset)
    print("Classifier Successfully Created...")
    print("Estimated Accuracy:")
    print(classifier_Accuracy)

    prediction_Result = prediction(classifier, processed_Input, new_features_for_training, column_names, label_names)


    return prediction_Result, input_Types





app = Flask(__name__)
app.config['UPLOADED_FILES_DEST'] = '/antenv/lib/python3.7/site-packages/flask/static/datasets'


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', title='Home')

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

            # Machine Learning declaration
            training_Input = os.path.join('static/datasets', train_filename)
            input_File = os.path.join('static/datasets', filename)


            prediction_Result, input_Types = main(input_File, training_Input)

            overall_Result = input_Types
            overall_Result = pd.concat([overall_Result, prediction_Result], axis=1, sort=True)
            overall_Result.columns = ['Actual Dataset', 'Prediction Result']

            return render_template('results.html', title='Results', Result=overall_Result)
    else:
        return render_template('home.html', title='Error')

if __name__ == '__main__':
    app.run(debug=True)
