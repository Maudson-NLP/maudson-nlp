import os
from flask import Flask
from flask import request
from summarizer import *


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
        src = os.getcwd()
        file.save(os.path.join(src, file.filename))

    summary = summarize(file.filename)

    return summary




if __name__ == '__main__':
    app.run() #host=HOST, port=PORT, debug=debug, threaded=threaded)
