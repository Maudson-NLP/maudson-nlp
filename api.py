import os
import json
import distutils
from flask import Flask
from flask import request
from summarizer import summarize
import keyword_extraction as kp


app = Flask(__name__, static_url_path='', static_folder='.')



@app.route('/summarize', methods=['GET', 'POST'])
def summarize_route():
    if 'file' not in request.files:
        return 'no file provided'
    file = request.files['file']
    if file.filename == '':
        return 'no file provided'
    if file:
        # filename = secure_filename(file.filename)
        src = os.getcwd() + '/uploaded_data/'
        file.save(os.path.join(src, file.filename))

    columns = request.form['columns'].split('%')
    l = int(request.form['l-value'])
    k = int(request.form['top-k'])
    form_use_bigrams = request.form['use-bigrams']
    form_use_svd = request.form['use-svd']
    form_use_noun_phrases = request.form['use-noun-phrases']

    import sys
    print(sys.version_info)
    print(help(distutils))

    use_bigrams = distutils.util.strtobool(form_use_bigrams)
    use_svd = distutils.util.strtobool(form_use_svd)
    use_noun_phrases = distutils.util.strtobool(form_use_noun_phrases)
    summary = summarize(file.filename, columns, l, use_bigrams, use_svd, k, use_noun_phrases=use_noun_phrases)

    return json.dumps(summary)



@app.route('/keyphrases', methods=['GET', 'POST'])
def return_keyphrases():
    
    print request.files
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
    if 'nb-keyphrases' not in request.files:
        nb_kp = 10
    else:
        nb_kp = request.files['nb-keyphrases']
    keyphrases = kp.extract_keyphrases('uploaded_data/'+file.filename, nb_kp)
    
    return json.dumps(keyphrases)



if __name__ == '__main__':
    env_port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=env_port, debug=True)
