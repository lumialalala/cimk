import pandas as pd
import numpy as np
import codecs
import yaml
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(0)


class Trainer(object):
    def __init__(self, conf):
        self.conf = conf
        self.file_re = conf["file_re"]
        self.file_ft_test = conf["file_ft_test"]
        self.file_ft_train = conf["file_ft_train"]
        self.file_ft_val = conf["file_ft_val"]
        self.file_ft_train_split = conf["file_ft_train_split"]

    def load_data(self, filepath):
        f = codecs.open(filepath, "r", "utf-8")
        data_list = []
        for line in f:
            data_list.append(line)
        f.close()
        return data_list

    def load_val_data(self):
        file_train = self.file_ft_train

        f_train = codecs.open(file_train, 'r', 'utf-8')
        train_list = []
        for eachline in f_train:
            train_list.append(eachline)
        f_train.close()

        # 训练集21400条数据，选择2000条作为验证集
        file_val = self.file_ft_val
        val_list = np.random.choice(train_list, 2000, replace=False)
        f_val = codecs.open(file_val, 'w', 'utf-8')
        for each in val_list:
            f_val.write(each)
        f_val.close()

        # 将去除验证集中之后的列表写入训练集
        file_ft_train_split = self.file_ft_train_split
        f_train = codecs.open(file_ft_train_split, 'w', 'utf-8')
        count_s=0
        count_n = 0
        for each in train_list:
            if each in val_list:
                count_s += 1
            else:
                f_train.write(each)
                count_n+=1
        f_train.close()
        print(count_s, count_n, count_n+count_s)

    # def train_test_split(self):
    #   X =
    #   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def test_model(self, model):
        output = open(self.file_re, "w")
        ft_test = codecs.open(self.file_ft_test, 'r', "utf-8")
        for eachline in ft_test:
            eachline = eachline.strip()
            re = model.predict(eachline, k=2)
            labels = re[0]
            probas = re[1]
            for i in range(0, 2):
                if labels[i] == '__label__1':
                    score = probas[i]
                    result = str(score) + "\n"
                    output.write(result)
        output.close()


if __name__ == '__main__':
    file_config = open("../../conf/v1.0/config.yaml")
    print("aaa")
    conf = yaml.load(file_config)
    trainer = Trainer(conf)
    trainer.load_val_data()
