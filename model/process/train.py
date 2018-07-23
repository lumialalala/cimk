from preprocess import preprocess
from trainer import Trainer
import yaml
import sys
sys.append("../define")
from fasttext import ft


def load_conf(file_config):
    file_conf = open(file_config)
    conf = yaml.load(file_conf)
    file_conf.close()
    return conf

def load_model(file_config):
    conf = load_conf(file_config)
    ft_model = ft(conf)
    model = ft_model.train()
    return model

def main(file_config):
    conf = load_conf(file_config)
    #pro = preprocess(conf)
    #pro.process()
    model = load_model(file_config)
    trainer = Trainer(model, conf)
    trainer.process(model)


file_config = "../../conf/v1.0/config.yaml"
main(file_config)
#if __name__ == "main()":
#    file_config = "../../conf/cofig.yaml"
#    main()
