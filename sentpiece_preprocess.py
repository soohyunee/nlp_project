import db_connect
import numpy as np
import torch
import torch.nn as nn
import sentencepiece as spm

def getdata():
    train = db_connect.get_data()    
    test = db_connect.get_data(test=True)
    return train, test

def train_sentencepiece():
    # https://paul-hyun.github.io/vocab-with-sentencepiece/
    # conduct when training needed
    corpus_file = '/kaggle/input/korean-wikipedia-dataset/kowiki.txt'
    prefix = 'kowiki'           # the mode name for save
    vocab_size = 10000          # + special tokens (7)
    spm.SentencePieceTrainer.train(
        f"--input={corpus_file} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
        " --model_type=bpe" +
        " --max_sentence_length=999999" + # 문장 최대 길이
        " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
        " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
        " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
        " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
        " --user_defined_symbols=[SEP],[CLS],[MASK]")
    print('**---------- training sentencepiece is done! ----------**')


def do_sentencepiece(vocab, text):
    pieces = vocab.encode_as_pieces(text)
    ids = vocab.encode_as_ids(text)
    return pieces, ids

def main(train_sentpiece=False):
    train, test = getdata()
    if train_sentpiece:
        train_sentencepiece()
    vocab_file = '/home/soohyun/nlp_project/data/kowiki.model'

    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)
    max_seq_len = 10
    train_file, test_file = [], []
    # only take 'ids' from do_sentencepiece
    for dic in train:
        res1 = do_sentencepiece(vocab, dic['doc'])[1]
        train_file.append(torch.tensor(res1))
        if len(res1) > max_seq_len:
            max_seq_len = len(res1)        

    for dic in test:
        res2 = do_sentencepiece(vocab, dic['doc'])[1]
        test_file.append(torch.tensor(res2))
        if len(res2) > max_seq_len:
            max_seq_len = len(res2)

    ##TODO: should I have to make a lookup table?
    print('max length:', max_seq_len)
    # print('trainnnn:', train_file[:3])
    train_file = nn.utils.rnn.pad_sequence(train_file, batch_first=True, padding_value=0)
    test_file = nn.utils.rnn.pad_sequence(test_file, batch_first=True, padding_value=0)
    # print('trtrtrtr:', train_file[:2])
    print('train file size:',train_file.size())
    #-------------------------------------------------------------------------
    #TODO: the problem is THE TYPE of data is a 'list'
    #TODO: then, what should the type of data will be have to returned??

    ###########33 unannotate when do embedding with positional embedding
    # train_file = emb_process(train_file)
    # test_file = emb_process(test_file)

    for idx,emb in enumerate(train_file):
        train[idx]['doc'] = emb

    for idx,emb in enumerate(test_file):
        test[idx]['doc'] = emb

    return train, test

# https://paul-hyun.github.io/transformer-01/
def emb_process(int_lookup):
    n_vocab = len(int_lookup)
    print('n_vocab:', n_vocab)
    d_hidden = 128
    n_embedding = nn.Embedding(n_vocab, d_hidden)

    input_embeds = n_embedding(int_lookup)
    print('input_embeds size:', input_embeds.size())
    return input_embeds
    

if __name__=="__main__":
    main()      # for test