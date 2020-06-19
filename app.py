from flask import Flask, render_template, request, redirect, jsonify
app = Flask(__name__)
import io
import json


import argparse
from const import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from utils import split, textprocess
import os
from metrics import f1_score_merged
from metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dataloader import DataLoader, Corpus, load_obj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import Transformer_Mix, get_attn_pad_mask
from metrics import get_entities
from evaluate import load_mask, softmax_mask


class DataLoader_test(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.word2idx = load_obj(self.save_dir + "dict.json")
        self.config = load_obj(self.save_dir + "Config.json")
        self.max_len = self.config["max_len"]

    def load_sentences(self, sent):
        """Loads sentences and tags from their corresponding files.
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentence = []

        tokens = split(textprocess(sent))
        sentence.append(self.convert_tokens_to_ids(tokens))
        return tokens, torch.tensor(sentence, dtype=torch.long)

    def convert_tokens_to_ids(self, tokens):
        sentence = []
        assert BOS == self.word2idx[WORD[BOS]]
        assert UNK == self.word2idx[WORD[UNK]]

        sentence.append(self.word2idx[WORD[BOS]])
        for tok in tokens:
            if tok in self.word2idx:
                sentence.append(self.word2idx[tok])
            else:
                sentence.append(self.word2idx[WORD[UNK]])
        pad = [self.word2idx[WORD[PAD]]]*(self.max_len+1 - len(sentence))

        assert len(sentence + pad) == self.max_len+1
        return sentence + pad

def get_prediction(model, sentence, save_dir, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    


    idx2lbl = load_obj(save_dir + "idx2lbl.json")
    idx2cls  = load_obj(save_dir + "idx2cls.json")
    
    enc = sentence.to(device)
    enc_self_attn_mask = get_attn_pad_mask(enc, enc)
    enc_self_attn_mask.to(device)

    # get results from model
    logits_tgt, logits_clsf = model(enc,enc_self_attn_mask)

    # get sentence length
    pad_num = enc.data.eq(0).sum(axis = 1)


    score_cls, cls_idx = torch.max(logits_clsf, dim = -1)
    pred_cls = cls_idx[0].data.tolist()



    # get valid slot for a specific intent
    idx_mask = load_mask(save_dir)


    masked_logits_tgt= softmax_mask(logits_tgt, cls_idx, idx_mask)
    score_tgt, tgt_idx = torch.max(masked_logits_tgt ,dim = -1)
    

    pred_tags = tgt_idx[0, 0:-pad_num].data.tolist()
    
    pred_lbls = []
    for idx in pred_tags:
        pred_lbls.append(idx2lbl[str(idx)])
    pred_cls = idx2cls[str(pred_cls)]

    
    return pred_cls ,pred_lbls

def pretty_print(tokens, pred_lbls, pred_cls):
    # ans = '\n==============RAW==================' +'\n'
    ans = ''
    # print('\n==============RAW==================', flush=True)
    ans += '{0}\n\r{1}'.format(' '.join(tokens), ' '.join(pred_lbls))+'\n\r'
    # print('{0}\n{1}'.format(' '.join(tokens), ' '.join(pred_lbls)), flush=True)

    chunks = get_entities(pred_lbls)
    slot_result = []
    # ans += '\n===================================' +'\n'
    # print('\n===================================', flush=True)
    ans += 'Intent\n\t'+'\n\r'
    # print('Intent\n\t', pred_cls, flush=True)
    ans += 'Slots'+'\n\r'
    # print('Slots', flush=True)
    for chunk in chunks:
        tag, start, end = chunk[0], chunk[1], chunk[2]
        tok = ''.join(tokens[chunk[1]:chunk[2]+1])
        string = '<{0}>: {1}'.format(tag, tok)
        slot_result.append(string)
    slot = '\t'+'\n\t'.join(slot_result)+'\n\r'
    # print('\t'+'\n\t'.join(slot_result), flush=True)
    # print('===================================', flush=True)
    return slot


@app.route('/query', methods=['GET'])
def predict():
    parser = argparse.ArgumentParser(description='Transformer NER')
    # parser.add_argument('--corpus-data', type=str, default='../data/auto_only-nav-distance_BOI.txt',
                        # help='path to corpus data')
    parser.add_argument('--save-dir', type=str, default='./data_char/',
                        help='path to save processed data')
    parser.add_argument('--pre-w2v', type=str, default='../data/w2v')
    args = parser.parse_args()


    pre_w2v = torch.load(args.save_dir + 'pre_w2v')
    pre_w2v = torch.Tensor(pre_w2v).to(device)

    model_ckpt = torch.load(os.path.join(args.save_dir, '{}.pyt'.format("Transformer_NER_best")),map_location=torch.device(device))

    config = load_obj(args.save_dir+'Config.json')
    model =Transformer_Mix(config, pre_w2v).to(device)
    model.load_state_dict (model_ckpt['model'])
    model.eval()


    data_loader = DataLoader_test(args.save_dir)

    input_sentence = request.args.get('text')
    tokens, test_data = data_loader.load_sentences(input_sentence)
    pred_cls ,pred_lbls = get_prediction(model, test_data, args.save_dir, mark='Test', verbose=True)
    slot = pretty_print(tokens, pred_lbls, pred_cls)
    return render_template('result.html', input_sentence = input_sentence,pred_cls=''.join(pred_cls), pred_lbls=' '.join(pred_lbls),slot=slot)
    # if request.method == 'POST':
    #     if 'inputsentence' not in request.files:
    #         return redirect(request.url)
    #     input_sentence = request.files.get('inputsentence')
    #     # input_sentence= "导航到世纪大道"
    #     tokens, test_data = data_loader.load_sentences(input_sentence)
    #     pred_cls ,pred_lbls = get_prediction(model, test_data, args.save_dir, mark='Test', verbose=True)
    #     return render_template('result.html', class_id=' '.join(pred_cls), class_name=' '.join(pred_lbls))
    #     # # ans = pretty_print(tokens, pred_lbls, pred_cls)
    # return render_template('index.html')

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
    # app.run(debug=True, port=5000, host='0.0.0.0') 
    app.run(debug=True, port=5000, host='0.0.0.0') #run app in debug mode on port 5000
