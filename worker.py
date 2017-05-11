import os
import nltk

import redis
from rq import Worker, Queue, Connection


listen = ['high', 'default', 'low']

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')

conn = redis.from_url(redis_url)



def error_handler(job, exc_type, exc_value, traceback):
    print("Worker Error: ")
    print(job)
    print(exc_type)
    print(exc_value)
    print(traceback)
    return True



if __name__ == '__main__':
    # Install nltk tools on Heroku
    if os.environ.get('HEROKU'):
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')

    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.push_exc_handler(error_handler)

        worker.work()

