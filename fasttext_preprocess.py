import fasttext
import argparse
from gensim.models import KeyedVectors

## use 'fasttext'
# https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
def get_fasttext(data_path):
    ft = fasttext.load_model(data_path)
    return ft

def get_data(data_path):
    with open(data_path) as f:
        ret = f.readlines()
    return ret[1:]

def incorporate_data(data):
    ret = []
    for id_doc_label in data:
        id_doc_label = id_doc_label.split("\t")
        tmp = {'id':0, 'doc':0, 'label':0}
        tmp['id'] = id_doc_label[0]
        tmp['doc'] = id_doc_label[1]
        tmp['label'] = id_doc_label[2].replace("\n","")
        ret.append(tmp)
    return ret
    
def main(args):
    wv = get_fasttext(args.pretrain)
    train_data = get_data(args.train)
    test_data = get_data(args.test)
    train = incorporate_data(train_data)
    test = incorporate_data(test_data)
    print('train data:', train[1])
    return train, test

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain',\
                        default = '/home/soohyun/nlp_project/data/fasttext/cc.ko.300.bin',\
                        help='PATH of .bin file contained pretrained word vectors')
    parser.add_argument('--train',\
                        default = '/home/soohyun/nlp_project/data/nsmc/ratings_train.txt',\
                        help='PATH of .txt file for training')
    parser.add_argument('--test',\
                        default = '/home/soohyun/nlp_project/data/nsmc/ratings_test.txt',\
                        help='PATH of .txt file for test')
    args = parser.parse_args()
    main(args)