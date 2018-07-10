import fasttext


#classifier=fasttext.supervised("../data/ft_trainset.txt","train_fasttext.model")
classifier=fasttext.load_model("train_fasttext.model.bin", label_prefix="__lable__")


df=pd.read_csv("../data/ft_testset.csv")

def train():
    output=open("./outputfile.csv","w")
    output.write("_id,context,y_true,y_pre,score")
    for index, row in df.iterrows():
            cid=row["_id"]
            context=row["context"]
            text=[]
            text.append(context)
            y_pre=re[0][0][0]
            score=re[0][0][1]
            output.write(cid+','+context+','+y_true+','+y_pre+','+score)
    output.close()