"""from django.shortcuts import render
import pandas as pd
import joblib


def home(request):
    # Your existing home function remains unchanged
    return render(request, 'home.html')

# Import necessary libraries
from django.shortcuts import render
import pandas as pd
import joblib

# Define the view function for the result page
def result(request):
    # Check if the request method is POST and the file is uploaded
    if request.method == 'POST' and request.FILES['file']:
        # Get the uploaded CSV file
        csv_file = request.FILES['file']

        # Check if the uploaded file is a CSV file
        if csv_file.name.endswith('.csv'):
            try:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(csv_file)
                df.drop("State",axis = 1,inplace=True)
                df.drop("Id",axis = 1,inplace=True)
                df["International plan"]=df["International plan"].replace({"Yes":1, "No":0})  # labelEncoding on International plan
                df["Voice mail plan"] = df["Voice mail plan"].replace({"Yes":1,"No":0})
                df["Area code"]=df["Area code"].replace({415:0,408:1,510:3})

                
                

                # Load the Telecom Churn Predictor model
                churn_predictor = joblib.load('tele.sav')

                # Make predictions using the loaded model
                churn_predictions = churn_predictor.predict(df)

                # Pass the predictions to the result template
                return render(request, "result.html", {'churn_predictions': churn_predictions})

            except Exception as e:
                # Handle exceptions and render error template
                return render(request, 'error.html', {'error_message': str(e)})
        else:
            # Render error template if the file format is invalid
            return render(request, 'error.html', {'error_message': 'Invalid file format. Please upload a CSV file.'})
    else:
        # Render error template if the request method is not POST or file is not uploaded
        return render(request, 'error.html', {'error_message': 'No file uploaded.'})


    return render(request, "result.html", {'churn_predictions': churn_predictions})
    return render(request, 'home.html')"""



"""from django.shortcuts import render
import pandas as pd
import joblib

def home(request):
    return render(request, 'home.html')

def result(request):
    if request.method == 'POST' and request.FILES['file']:
        # Load the Telecom Churn Predictor model
        churn_predictor = joblib.load('tele.sav')

        # Load CSV file
        csv_file = request.FILES['file']
        if csv_file.name.endswith('.csv'):
            # Specify encoding as 'latin1' or 'ISO-8859-1'
            df = pd.read_csv(csv_file, encoding='latin1')

            # Make predictions
            churn_predictions = churn_predictor.predict(df)

            return render(request, "result.html", {'churn_predictions': churn_predictions})
        else:
            return render(request, 'error.html', {'error_message': 'Invalid file format. Please upload a CSV file.'})

    return render(request, 'home.html')"""

    # Import necessary libraries
from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render

"""def home(request):
    return render(request, 'home.html')

# Define the view function for the result page
def result(request):
    # Check if the request method is POST and the file is uploaded
    if request.method == 'POST' and request.FILES['file']:
        # Get the uploaded CSV file
        csv_file = request.FILES['file']

        # Check if the uploaded file is a CSV file
        if csv_file.name.endswith('.csv'):
            try:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(csv_file)
                df.drop("State",axis = 1,inplace=True)
                df.drop("Id",axis = 1,inplace=True)
                df["International plan"]=df["International plan"].replace({"Yes":1, "No":0})  # labelEncoding on International plan
                df["Voice mail plan"] = df["Voice mail plan"].replace({"Yes":1,"No":0})
                df["Area code"]=df["Area code"].replace({415:0,408:1,510:3})

                # Load the Telecom Churn Predictor model
                churn_predictor = joblib.load('tele.sav')

                # Make predictions using the loaded model
                # Assuming that your model expects specific columns for prediction
                churn_features = ['Account length', 'Area code', 'International plan', 'Voice mail plan',
                                  'Number vmail messages', 'Total day minutes', 'Total day calls',
                                  'Total day charge', 'Total eve minutes', 'Total eve calls',
                                  'Total eve charge', 'Total night minutes', 'Total night calls',
                                  'Total night charge', 'Total intl minutes', 'Total intl calls',
                                  'Total intl charge', 'Customer service calls']
        for index, row in df.iterrows():
            # Prepare data for prediction
            churn_predictor = joblib.load('tel.sav')

            # Extract features for prediction
            churn_features = row[['Account length', 'Area code', 'International plan', 'Voice mail plan',
                                'Number vmail messages', 'Total day minutes', 'Total day calls',
                                'Total day charge', 'Total eve minutes', 'Total eve calls',
                                'Total eve charge', 'Total night minutes', 'Total night calls',
                                'Total night charge', 'Total intl minutes', 'Total intl calls',
                                'Total intl charge', 'Customer service calls']]

            # Reshape the data for prediction
            churn_features_reshaped = churn_features.values.reshape(1, -1)

            # Create a DataFrame for prediction
            churn_prediction_df = pd.DataFrame(churn_features_reshaped, columns=churn_features.index)

            # Make prediction
            churn_prediction = churn_predictor.predict(churn_prediction_df)

            # Construct the prediction dictionary
            prediction_dict = {
                'Account length': row['Account length'],
                'Area code': row['Area code'],
                'International plan': row['International plan'],
                'Voice mail plan': row['Voice mail plan'],
                'Number vmail messages': row['Number vmail messages'],
                'Total day minutes': row['Total day minutes'],
                'Total day calls': row['Total day calls'],
                'Total day charge': row['Total day charge'],
                'Total eve minutes': row['Total eve minutes'],
                'Total eve calls': row['Total eve calls'],
                'Total eve charge': row['Total eve charge'],
                'Total night minutes': row['Total night minutes'],
                'Total night calls': row['Total night calls'],
                'Total night charge': row['Total night charge'],
                'Total intl minutes': row['Total intl minutes'],
                'Total intl calls': row['Total intl calls'],
                'Total intl charge': row['Total intl charge'],
                'Customer service calls': row['Customer service calls'],
                'Prediction': churn_prediction[0]
            }

            # Store the prediction
            predictions.append(prediction_dict)

                
                # Ensure that the columns in the DataFrame match the features expected by the model
                if set(churn_features).issubset(df.columns):
                    # Make predictions using the loaded model
                    churn_predictions = churn_predictor.predict(df[churn_features])

                    # Pass the predictions to the result template
                    return render(request, "result.html", {'churn_predictions': churn_predictions})
                else:
                    # Handle case where DataFrame columns do not match expected features
                    error_message = 'DataFrame columns do not match expected features for prediction.'
                    return render(request, 'error.html', {'error_message': error_message})

            except Exception as e:
                # Handle exceptions and render error template
                return render(request, 'error.html', {'error_message': str(e)})
        else:
            # Render error template if the file format is invalid
            return render(request, 'error.html', {'error_message': 'Invalid file format. Please upload a CSV file.'})
    else:
        # Render error template if the request method is not POST or file is not uploaded
        return render(request, 'error.html', {'error_message': 'No file uploaded.'})

    return render(request, "result.html", {'churn_predictions': churn_predictions})"""
import pandas as pd
from django.shortcuts import render
import joblib
import numpy as np

def home(request):
    return render(request, 'home.html')

import pandas as pd
from django.shortcuts import render
import joblib

def result(request):
    if request.method == 'POST' and request.FILES.get('file'):
        csv_file = request.FILES['file']
        if csv_file.name.endswith('.csv'):
            try:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(csv_file)

                # Check if the DataFrame has columns
                if df.empty:
                    return render(request, 'error.html', {'error_message': 'No data in the CSV file.'})
                ids = df['Id']
                states = df['State']

                # Data preprocessing and feature engineering
                # Drop unnecessary columns

                #df.drop(["State", "Id"], axis=1, inplace=True)

                # Label encoding on 'International plan' and 'Voice mail plan'
                df["International plan"] = df["International plan"].replace({"Yes": 1, "No": 0})
                df["Voice mail plan"] = df["Voice mail plan"].replace({"Yes": 1, "No": 0})

                # Replace 'Area code' values
                df["Area code"] = df["Area code"].replace({415: 0, 408: 1, 510: 3})

                # Load the Telecom Churn Predictor model
                churn_predictor = joblib.load('tele.sav')

                predictions = []

                # Iterate over each row for prediction
                for index, row in df.iterrows():
                    # Prepare data for prediction
                    churn_features = row[['Account length', 'Area code', 'International plan', 'Voice mail plan',
                                          'Number vmail messages', 'Total day minutes', 'Total day calls',
                                          'Total day charge', 'Total eve minutes', 'Total eve calls',
                                          'Total eve charge', 'Total night minutes', 'Total night calls',
                                          'Total night charge', 'Total intl minutes', 'Total intl calls',
                                          'Total intl charge', 'Customer service calls']]

                    churn_features_reshaped = churn_features.values.reshape(1, -1)

                    # Create a DataFrame for prediction
                    churn_prediction_df = pd.DataFrame(churn_features_reshaped, columns=churn_features.index)

                    # Make prediction
                    churn_prediction = churn_predictor.predict(churn_prediction_df)

                    # Construct the prediction dictionary
                    # prediction_dict = {
                    #     'Account length': row['Account length'],
                    #     'Area code': row['Area code'],
                    #     'International plan': row['International plan'],
                    #     'Voice mail plan': row['Voice mail plan'],
                    #     'Number vmail messages': row['Number vmail messages'],
                    #     'Total day minutes': row['Total day minutes'],
                    #     'Total day calls': row['Total day calls'],
                    #     'Total day charge': row['Total day charge'],
                    #     'Total eve minutes': row['Total eve minutes'],
                    #     'Total eve calls': row['Total eve calls'],
                    #     'Total eve charge': row['Total eve charge'],
                    #     'Total night minutes': row['Total night minutes'],
                    #     'Total night calls': row['Total night calls'],
                    #     'Total night charge': row['Total night charge'],
                    #     'Total intl minutes': row['Total intl minutes'],
                    #     'Total intl calls': row['Total intl calls'],
                    #     'Total intl charge': row['Total intl charge'],
                    #     'Customer service calls': row['Customer service calls'],
                    #     'Prediction': churn_prediction[0]
                    # }

                     # Create a dictionary to store dataset values and prediction
                    prediction_dict = {
                        'Id': ids[index],
                        'State': states[index],
                        'Prediction': churn_prediction[0]
                    }

                    # Append the prediction dictionary to the predictions list
                    predictions.append(prediction_dict)

                    # # Append the prediction dictionary to the predictions list
                    # predictions.append(prediction_dict)

                # Pass the predictions to the result template
                return render(request, "result.html", {'predictions': predictions})

            except Exception as e:
                return render(request, 'error.html', {'error_message': str(e)})

        else:
            return render(request, 'error.html', {'error_message': 'Invalid file format. Please upload a CSV file.'})

    else:
        return render(request, 'error.html', {'error_message': 'No file uploaded.'})
