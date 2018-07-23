#import fastText as fasttext
import fasttext

class ft(object):
    def __init__(self, conf):
        self.conf = conf
        self.file_ft_train = conf["file_ft_train"]
        #self.file_es_vec = conf["file_es_vec"]
        # , pretrainedVectors=self.file_es_vec
    def train(self):
        classifier = fasttext.train_supervised(input=self.file_ft_train, label = "__label__",ws = 10, epoch = 25)
        return classifier

