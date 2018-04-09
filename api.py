import os
import uuid
import nltk
import json
# import tinys3
from flask import Flask
from flask import request, send_file, abort
from summarizer import summarize
import pandas as pd
import keyword_extraction as kp
# from rq import Queue
# from worker import conn


app = Flask(__name__, static_url_path='', static_folder='.')
# Set up the worker Queue
# q = Queue(connection=conn)
#AWS S3
# keyId = os.environ['S3_KEY']
# sKeyId= os.environ['S3_SECRET']
# conn = tinys3.Connection(keyId, sKeyId, tls=True)


@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/index.html')
def rootbis():
    return app.send_static_file('index.html')


@app.route('/summary_result', methods=['POST'])
def summary_result():
    result_id = request.form['result_id']
    try:
        # c = conn.get(result_id + '.json', bucket='clever-nlp')
        c = open(result_id + '.json')
        return json.dumps(json.loads(c.read()))
    except Exception as e:
        print(e)
        abort(404)


@app.route('/summarize', methods=['POST'])
def summarize_route():
    """
    Flask route for summarization + noun phrases task
    :return: Json summary response
    """
    src = os.getcwd()
    if len(request.files):
        file = request.files['file']
        filename = file.filename.replace('+', '')
        file_full = os.path.join(src, filename)
        file.save(file_full)

        f = open(file_full, 'rb')
        # conn.upload(filename, f, 'clever-nlp', public=False)
    else:
        textToSummarize = request.form['textToSummarize']
        filename = textToSummarize[:20]
        file_full = os.path.join(src, filename)
        with open(file_full, 'wb') as f:
            f.write(textToSummarize.encode('utf-8').strip())
        f = open (file_full, 'rb')
        # conn.upload(filename, f, 'clever-nlp', public=False)


    if request.form['columns']:
        columns = request.form['columns'].split('%')
    else:
        columns = []

    l = int(request.form['l-value'])
    k = int(request.form['top-k'])
    form_ngram_min = request.form['ngram-min']
    form_ngram_max = request.form['ngram-max']

    form_use_svd = request.form['use-svd']
    form_tfidf = request.form['tfidf']
    form_scale_vectors = request.form['scale-vectors']
    form_use_noun_phrases = request.form['use-noun-phrases']
    form_split_longer_sentences = request.form['split-longer-sentences']
    form_split_length = request.form['to-split-length']
    form_group_by = request.form['group-by']
    form_extract_sibling_sents = request.form['extract-sibling-sents']
    form_exclude_misspelled = request.form['exclude-misspelled']

    ngram_min = int(form_ngram_min)
    ngram_max = int(form_ngram_max)
    ngram_range = (ngram_min, ngram_max)

    use_svd = strtobool(form_use_svd)
    tfidf = strtobool(form_tfidf)
    scale_vectors = strtobool(form_scale_vectors)
    use_noun_phrases = strtobool(form_use_noun_phrases)
    split_longer_sentences = strtobool(form_split_longer_sentences)
    extract_sibling_sents = strtobool(form_extract_sibling_sents)
    exclude_misspelled = strtobool(form_exclude_misspelled)

    summary_id = str(uuid.uuid4())

    # q.enqueue(
    summarize(
        summary_id,
        filename,
        columns=columns,
        group_by=form_group_by,
        l=l,
        ngram_range=ngram_range,
        tfidf=tfidf,
        use_svd=use_svd,
        k=k,
        scale_vectors=scale_vectors,
        use_noun_phrases=use_noun_phrases,
        split_longer_sentences=split_longer_sentences,
        to_split_length=int(form_split_length),
        extract_sibling_sents=extract_sibling_sents,
        exclude_misspelled=exclude_misspelled
    )

    return summary_id



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
    
    
    groupby = request.form['groupby']
    headers = request.form['headers']
    nb_kp = request.form['nb_keyphrases']
    min_char_length = request.form['min_char_length']
    max_words_length = request.form['max_words_length']
    min_words_length = request.form['min_words_length']
    min_keyword_frequency = request.form['min_keyword_frequency']
    tradeoff = request.form['tradeoff']
 
    if len(groupby) == 0:
        keyphraz = kp.extract_keyphrases_survey(
                                               filename='uploaded_data/'+file.filename,
                                               nb_kp=nb_kp,
                                               min_char_length=min_char_length,
                                               max_words_length=max_words_length,
                                               min_words_length=min_words_length,
                                               min_keyword_frequency=min_keyword_frequency,
                                               groupby=groupby,
                                               headers=headers,
                                               tradeoff=tradeoff)
         
    elif len(groupby) != 0:
        keyphraz = kp.extract_keyphrases_reviews(
                                                filename='uploaded_data/'+file.filename,
                                                nb_kp=nb_kp,
                                                min_char_length=min_char_length,
                                                max_words_length=max_words_length,
                                                min_words_length=min_words_length,
                                                min_keyword_frequency=min_keyword_frequency,
                                                groupby=groupby,
                                                headers=headers,
                                                tradeoff=tradeoff)
    
    pd.DataFrame(keyphraz).to_csv('static/keyphrases.csv')
    
    return json.dumps(keyphraz)



@app.route('/export')
def export():
    return send_file('static/keyphrases.csv', attachment_filename='keyphrases.csv')



def strtobool (val):
    """
    Copied from https://github.com/python-git/python/blob/master/Lib/distutils/util.py
    Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = str.lower(str(val))
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise(ValueError, "invalid truth value %r" % (val,))



if __name__ == '__main__':
    # Install nltk tools on Heroku
    if os.environ.get('HEROKU'):
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')

    env_port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=env_port, debug=True)
