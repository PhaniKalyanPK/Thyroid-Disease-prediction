import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random
app = Flask(__name__) #Initialize the flask App

model_rf = pickle.load(open("Random_forest_model.pkl", "rb"))
model_xgb = pickle.load(open("XGBoost_model.pkl", "rb"))


@app.route('/')

@app.route('/index')
def index():
	return render_template('index.html')





@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('age', inplace=True)
        return render_template("preview.html",df_view = df)	


#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')



@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    features_np = [np.array(features)]
    pred_1 = model_rf.predict(features_np)
    pred_2 = model_xgb.predict(features_np)

    prediction = (pred_1+pred_2)/2 
    prediction = np.where(prediction>=0.49, 1, 0)
    
    if prediction == 0:
        result = "Good News! You are free from thyroidal disease."
    elif prediction == 1:
       result = "Our model has predicted that you have thyroidal disease."
        
    return render_template("prediction.html", prediction_text=result) 

    
if __name__ == "__main__":
    app.run(debug=True)
