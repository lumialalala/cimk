#import fastText as fasttext
import fasttext

class ft(object):
    def __init__(self, conf):
        self.conf = conf
        self.file_ft_train = conf["file_ft_train"]
        self_file_ft_test = conf["file_ft_test"]
    def train(self):
        classifier = fasttext.train_supervised(input=self.file_ft_train, label = "__label__",ws = 8, epoch = 10)
        return classifier

