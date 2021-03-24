import db_connect
import numpy as np
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
        vocab_file = '/home/soohyun/nlp_project/kowiki.model'
    else:
        vocab_file = '/home/soohyun/nlp_project/kowiki.model'

    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)

    # only take 'ids' from do_sentencepiece
    for idx,dic in enumerate(train):
        train[idx]['doc'] = do_sentencepiece(vocab, dic['doc'])[1]

    for idx,dic in enumerate(test):
        test[idx]['doc'] = do_sentencepiece(vocab, dic['doc'])[1]

    print('train:', train[:3])

# def preprocess(dict_list):
    # df.drop_duplicates(subset=['doc'], inplace=True)
    # df.dropna(how='any')
    # return dict_list

main()