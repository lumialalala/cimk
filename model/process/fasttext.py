#import fastText as fasttext
import fasttext

class ft(object):
    def __init__(self, conf):
        self.conf = conf
        self.file_ft_train = conf["file_train"]
        self_file_ft_test = conf["file_ft_test"]
    def train(self):
        classifier = fasttext.supervised(input=self.file_ft_train, label_prefix="__lable__",ws = 8, epoch = 10)
        return classifier

