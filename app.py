from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    '''
    for rendering results on html GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction
    
    if output == 1:
        return render_template('index.html',prediction_text = "Output - He/She Will Donate Blood" )
    else:
        return render_template('index.html',prediction_text = "Output - He/She Will Not Donate Blood" )
                               

  
#app.route('/predict_api',methods=['POST'])
#ef predict_api():
   #'''
  # For direct API calls trought request
 #  '''
 #  data = request.get_json(force=True)
#   prediction = model.predict([np.array(list(data.values()))])

if __name__ == "__main__":
    app.run(debug = True)