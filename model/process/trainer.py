class Trainer(object):
    def __init__(self, model, conf):
        self.conf = conf
        self.file_re = conf["file_re"]
        self.file_ft_test = conf["file_ft_test"]

    def test_model(self, model):
        output = open(self.file_re, "w")
        ft_test = open(self.file_ft_test)
        for eachline in ft_test:
            eachline = eachline.strip()
            re = model.predict(eachline, k=2)
            labels = re[0]
            probas = re[1]
            for i in range(0, 2):
                if labels[i] == '__label__1':
                    score = probas[i]
                    result = str(score)+"\n"
                    output.write(result)
        output.close()
