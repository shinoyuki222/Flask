import argparse
from const import *
import numpy as np
from utils import load_obj,split,textprocess
import os
import unicodedata
import onnxruntime
from metrics import get_entities



def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size = 1
    len_q = len(seq_q[0])
    len_k = len(seq_k[0])
    # eq(zero) is PAD token
    pad_attn_mask = np.array(seq_k)==0  # (batch_size, 1, len_k/len_q) one is masking
    pad_mask = np.repeat(pad_attn_mask,len_q,axis=0).reshape(1,len_q,len_k)
    return pad_mask
    # return pad_attn_mask.repeat(batch_size, len_q, len_k)  # (batch_size, len_q, len_k)


def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x) 
        
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator =  1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
        
    assert x.shape == orig_shape
    return x

def softmax_mask(logits_tgt, cls_idx, idx_mask):
    logits_tgt = logits_tgt[0,:,:]
    mask = idx_mask[str(cls_idx)]
    length, tgt_num = logits_tgt.shape[0], logits_tgt.shape[1]
    scores_exp = softmax(logits_tgt)
     # this step masks
    scores_exp = scores_exp * mask
    return scores_exp


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
        return tokens, sentence

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



def test(ort_session, sentence, save_dir, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""

    idx2lbl = load_obj(save_dir + "idx2lbl.json")
    idx2cls  = load_obj(save_dir + "idx2cls.json")
    
    enc = sentence
    enc_self_attn_mask = get_attn_pad_mask(enc, enc)
    x = (enc,enc_self_attn_mask)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[i].name: x[i] for i in range(len(x))}
    logits_tgt, logits_clsf = ort_session.run(None, ort_inputs)

    pad_num = np.count_nonzero(enc)

    pred_cls = np.argmax(logits_clsf)
    score = logits_clsf[0,pred_cls]

    # get valid slot for a specific intent
    idx_mask = load_obj(save_dir + "idx_mask_onnx.json")


    masked_logits_tgt= softmax_mask(logits_tgt, pred_cls, idx_mask)
    tgt_idx = np.argmax(masked_logits_tgt, axis = 1)
    score_tgt = 0
    
    pred_tags = tgt_idx[:pad_num]
    
    pred_lbls = []
    for idx in pred_tags:
        pred_lbls.append(idx2lbl[str(idx)])
    pred_cls = idx2cls[str(pred_cls)]
    
    return pred_cls ,pred_lbls

def merged_slot(tokens, pred_lbls):
    chunks = get_entities(pred_lbls)
    slot_result = []
    for chunk in chunks:
        tag, start, end = chunk[0], chunk[1], chunk[2]
        tok = ''.join(tokens[chunk[1]:chunk[2]+1])
        string = '<{0}>: {1}'.format(tag, tok)
        slot_result.append(string)
    return slot_result

if __name__ == '__main__':
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

    # read test_sentence
    input_sentence = '导航到世纪大道一百一十八号'
    tokens, test_data = data_loader.load_sentences(input_sentence)
    
    # run inference
    pred_cls ,pred_lbls = test(ort_session, test_data, args.save_dir, mark='Test', verbose=True)

    # merge_slot
    slot = merged_slot(tokens, pred_lbls)

    # print results
    print(input_sentence)
    print(' '.join(pred_lbls))
    print(''.join(pred_cls))
    
    print('\n'.join(slot))


