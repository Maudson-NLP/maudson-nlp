import os
import json
from flask import Flask
from flask import request
from summarizer import *
import keyword_extraction as kp


app = Flask(__name__, static_url_path='', static_folder='.')



@app.route('/summarize', methods=['GET', 'POST'])
def hello_world():

    if 'file' not in request.files:
        return 'no file provided'
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return 'no file provided'
    if file:
        # filename = secure_filename(file.filename)
        src = os.getcwd() + '/uploaded_data/'
        file.save(os.path.join(src, file.filename))

    columns = request.form['columns'].split('%')

    summary = summarize(file.filename, columns)
    
    return json.dumps(summary)



@app.route('/keyphrases', methods=['GET', 'POST'])
def return_keyphrases():

    if 'file-keyphrase' not in request.files:
        return 'no file provided'
    file = request.files['file-keyphrase']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return 'no file provided'
    if file:
        # filename = secure_filename(file.filename)
        src = os.getcwd() + '/uploaded_data/'
        file.save(os.path.join(src, file.filename))

    keyphrases = kp.extract_keyphrases(file.filename)
    
    return json.dumps(keyphrases)



if __name__ == '__main__':
    # summary = summarize('survey_data.xlsx', ['What is Healthy Skin?', 'How do you know your skin is healthy?'])
    app.run(debug=True) #host=HOST, port=PORT, debug=debug, threaded=threaded)
