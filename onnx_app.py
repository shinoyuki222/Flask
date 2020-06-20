from flask import Flask, render_template, request, redirect, jsonify
app = Flask(__name__)
from load_onnx import DataLoader_test, test,merged_slot
import argparse
import os
import onnxruntime
from metrics import get_entities

parser = argparse.ArgumentParser(description='Transformer NER')
# parser.add_argument('--corpus-data', type=str, default='../data/auto_only-nav-distance_BOI.txt',
                    # help='path to corpus data')
parser.add_argument('--save-dir', type=str, default='./data_char/',
                    help='path to save processed data')
args = parser.parse_args()

# Init
data_loader = DataLoader_test(args.save_dir)
model = "transformer_mix.onnx"
ort_session = onnxruntime.InferenceSession(args.save_dir+ model)


@app.route('/query', methods=['GET'])
def predict():
    # read test_sentence
    input_sentence = request.args.get('text')
    tokens, test_data = data_loader.load_sentences(input_sentence)
    
    # run inference
    pred_cls ,pred_lbls = test(ort_session, test_data, args.save_dir, mark='Test', verbose=True)

    # merge_slot
    slot = merged_slot(tokens, pred_lbls)

    # print results
    out_lbls = ' '.join(pred_lbls)
    out_cls = ''.join(pred_cls)
    out_slot = ''.join(slot)

    return render_template('result.html', input_sentence = input_sentence,pred_cls=out_cls, pred_lbls=out_lbls,slot=out_slot)


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
