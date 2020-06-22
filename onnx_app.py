from flask import Flask, jsonify,request
app = Flask(__name__)
from load_onnx import NLU_module
import argparse
import os
import onnxruntime
from metrics import get_entities

Module = NLU_module()

@app.route('/query', methods=['GET'])
def predict():
    # read test_sentence
    input_sentence = request.args.get('text')
    results = Module.Inference(input_sentence)

    return jsonify(results)


@app.route('/query-example')
def query_example():
    input_sentence = request.args.get('text')
    return input_sentence

@app.route('/form-example')
def formexample():
    return 'Todo...'

@app.route('/json-example')
def jsonexample():
    return 'Todo...'

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0') 
    # app.run(debug=True, port=5000) #run app in debug mode on port 5000
