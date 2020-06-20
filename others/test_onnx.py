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
from model_onnx import Transformer_Mix, get_attn_pad_mask

from metrics import get_entities

from evaluate import load_mask, softmax_mask

def load_mask(save_dir):
    config = load_obj(save_dir + "Config.json")
    num_label = config['num_label']
    idx2lbl = load_obj(save_dir + "idx2lbl.json")
    idx2cls  = load_obj(save_dir + "idx2cls.json")
    dict_lbl = load_obj(save_dir + "dict_lbl.json")
    dict_clsf  = load_obj(save_dir + "dict_clsf.json")
    lbl_mask = load_obj(save_dir + "lbl_mask.json")

    idx_mask = {}
    for intent in lbl_mask:
        valid_slot = lbl_mask[intent]
        mask = [0] * num_label
        for s in valid_slot:
            idx = dict_lbl[s]
            mask[idx] = 1
        idx_mask[dict_clsf[intent]] = mask
        # print(idx_mask)
    
    return idx_mask

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

def test(model, sentence, save_dir, mark='Eval', verbose=False):
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
    print('\n==============RAW==================', flush=True)
    print('{0}\n{1}'.format(' '.join(tokens), ' '.join(pred_lbls)), flush=True)

    chunks = get_entities(pred_lbls)
    slot_result = []
    print('\n===================================', flush=True)
    print('Intent\n\t', pred_cls, flush=True)
    print('Slots', flush=True)
    for chunk in chunks:
        tag, start, end = chunk[0], chunk[1], chunk[2]
        tok = ''.join(tokens[chunk[1]:chunk[2]+1])
        string = '<{0}>: {1}'.format(tag, tok)
        slot_result.append(string)
    
    print('\t'+'\n\t'.join(slot_result), flush=True)
    print('===================================', flush=True)

if __name__ == '__main__':

    print('device = ', device, flush=True)
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

    # Initialize the DataLoader
    data_loader = DataLoader_test(args.save_dir)

    print("Starting test...", flush=True)
    print('Please add a space between English and Chinese', flush=True)


    input_sentence = '导航到世纪大道一百一十八号'
    # Check if it is quit case
    if input_sentence == 'q' or input_sentence == 'quit': exit()
    # Normalize sentence
    tokens, test_data = data_loader.load_sentences(input_sentence)
    # print(input_sentence)
    # Evaluate sentence
    pred_cls ,pred_lbls = test(model, test_data, args.save_dir, mark='Test', verbose=True)
    # Format and print response sentence
    # pred_tags[:] = [x for x in pred_tags if x != 'PAD']
    print('{0}\n{1}'.format(input_sentence, ' '.join(pred_lbls)), flush=True)
    # print('intent:', pred_cls)
    # pretty_print(tokens, pred_lbls, pred_cls)
    # exit()

    import onnx
    model = model
    enc = test_data.to(device)
    enc_self_attn_mask = get_attn_pad_mask(enc, enc)
    enc_self_attn_mask.to(device)
    x = (enc,enc_self_attn_mask)
    # get results from model
    logits_tgt, logits_clsf = model(enc, enc_self_attn_mask)
    # logits_tgt, logits_clsf = model(enc,enc_self_attn_mask)
    torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "transformer_mix.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'])

    import onnx

    onnx_model = onnx.load("transformer_mix.onnx")
    onnx.checker.check_model(onnx_model)


    import onnxruntime

    ort_session = onnxruntime.InferenceSession("transformer_mix.onnx")


    def to_numpy(tensor):
        # return tensor.cpu().numpy()
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()



    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[i].name: to_numpy(x[i]) for i in range(len(x))}
    ort_outs= ort_session.run(None, ort_inputs)




    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(logits_tgt), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(logits_clsf), ort_outs[1], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        
