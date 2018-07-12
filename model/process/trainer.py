class Trainer(object):
    def __init__(self, model, conf):
        self.conf = conf
        self.file_re = conf["file_re"]
        self.file_ft_test = conf["file_ft_test"]

    def test_model(self, model):
        output = open(self.file_re, "w")
        output.write("_id,context,y_true,y_pre,score")
        ft_test = open(self.file_ft_test)
        for eachline in ft_test:
            eachline = eachline.strip()
            re = model.predict(eachline, k=2)
            print(eachline, re)
            for i in range(0, 2):
                if re[0][i][0] == '1':
                    score = re[0][i][1]
                    y_pre = '1'
                    result = str(score) + '\n'
                    output.write(result)
        output.close()
