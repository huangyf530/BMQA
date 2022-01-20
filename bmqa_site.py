from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from model_launcher import ModelLauncher
import os
from queue import Queue
import time
import threading

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model_laucher = None
args_file = 'configs/model_laucher/xl.json'
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
request_queue = Queue(maxsize=100)

@app.route('/')
def hello():
    return 'Hello World!'

@cross_origin()
@app.route('/query', methods=['GET'])
def query():
    question = request.args.get('question')
    request_queue.put(question)
    ans_predictions, no_ans_predictions = model_laucher.generate_answer(question)
    ans_predictions.extend(no_ans_predictions)
    request_queue.get()
    return str(ans_predictions)

@app.route('/query_html', methods=['GET'])
def query_html():
    question = request.args.get('question')
    print("Find answer for question:", question)
    ans_predictions, no_ans_predictions = model_laucher.generate_answer(question)
    ans_predictions.extend(no_ans_predictions)
    for index, p in enumerate(ans_predictions):
        p['index'] = index + 1
    return render_template("query.html", question=question, predictions=ans_predictions)

if __name__ == '__main__':
    # global model_laucher
    model_laucher = ModelLauncher(args_file)
    app.run(host='0.0.0.0', port=8890, threaded=False)