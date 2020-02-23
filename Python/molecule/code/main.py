from pandas import DataFrame, read_csv
from torch import device,randn,abs,save,load
from torch.nn import Linear, Module, MSELoss, Sequential, Sigmoid,Parameter
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from random import choices

class MyDatasets(Dataset):
    def __init__(self,data:DataFrame):
        super(MyDatasets,self).__init__()
        self.names=data.index.to_numpy().astype(str)
        datas=data.iloc[:,:-6]
        values=data.iloc[:,-6:]
        self.datas=datas.to_numpy().astype(float)
        self.values=values.to_numpy().astype(float)
    
    def __getitem__(self,index):
        name=self.names[index]
        data=self.datas[index]
        value=self.values[index]
        return name,data,value

    def __len__(self):
        return self.names.size


class MyModel(Module):
    def __init__(self,in_features:int):
        super(MyModel,self).__init__()
        self.mean_in=Parameter(randn(1,in_features))
        self.std_in=Parameter(randn(1,in_features))
        self.mean_out=Parameter(randn(1,6))
        self.std_out=Parameter(randn(1,6))
        self.encode=Sequential(
            Linear(in_features,2048),
            Sigmoid(),
            Linear(2048,1024),
            Sigmoid(),
            Linear(1024,256),
            Sigmoid()
        )
        self.decode=Sequential(
            Linear(256,1024),
            Sigmoid(),
            Linear(1024,2048),
            Sigmoid(),
            Linear(2048,in_features),
            Sigmoid()
        )
        self.features=Sequential(
            Linear(256,1024),
            Sigmoid(),
            Linear(1024,2048),
            Sigmoid(),
            Linear(2048,2048),
            Sigmoid(),
            Linear(2048,256),
            Sigmoid(),
            Linear(256,6),
            Sigmoid()
        )
    def forward_en(self,input):
        input=(input-self.mean_in)/self.std_in
        x=self.encode(input)
        return x

    def forward_de(self,input):
        x=self.decode(input)
        output=x*self.std_in+self.mean_in
        return output

    def forward_feature(self,input):
        output=self.features(input)*self.std_out+self.mean_out
        return output

    def forward(self,input):
        x=self.forward_en(input)
        out=self.forward_feature(x)
        return out

if __name__ == "__main__":
    redivide_data=False
    continue_train=True
    print("Reading from disk...")
    dataset=read_csv("train.csv")
    dataset=dataset.set_index(dataset['id'])
    print("Deleting useless data...")
    cdt=dataset.std()>0
    cdt=cdt[cdt]
    dataset=dataset[cdt.index]
    print("Dividing data sets")
    if redivide_data:
        continue_train=False
        total=len(dataset)
        len_validate=int(total/5)
        len_train=total-len_validate
        validate=choices(range(total),k=len_validate)
        train=[i for i in range(total) if i not in validate]
        print("Saving datasets...")
        data={
            "total":total,
            "train":train,
            "validate":validate,
            "len_train":len_train,
            "len_validate":len_validate
        }
        save(data,"datasets")
    else:
        data=load('datasets')
        train=data["train"]
        validate=data['validate']
        len_train=data['len_train']
        len_validate=data['len_validate']
    train=dataset.iloc[train]
    validate=dataset.iloc[validate]
    len_features=len(train.columns)-6
    loader_t=DataLoader(MyDatasets(train),batch_size=16)
    loader_v=DataLoader(MyDatasets(validate),batch_size=16)
    lr=0.001
    rule=MSELoss()
    cuda=device('cuda:0')
    model=MyModel(len_features).to(cuda)
    
    epochs=300
    writer=SummaryWriter(log_dir='./run')
    print('Start epochs...')
    epoch=1
    if continue_train:
        checkpoint=load('checkpoint')
        epoch=checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model'])
    opt=SGD(model.parameters(),lr=lr)
    while epoch <= epochs:
        print("Train...")
        model.train()
        loss_de=0.0
        loss_pr=0.0
        for _,data,value in loader_t:
            data=data.float().to(cuda)
            value=value.float().to(cuda)
            encode=model.forward_en(data)
            decode=model.forward_de(encode)
            output=model.forward_feature(encode)
            loss_decode=rule(decode,data)
            loss_predict=rule(output,value)
            loss_de+=loss_decode.item()
            loss_pr+=loss_predict.item()
            loss=loss_decode+loss_predict
            opt.zero_grad()
            loss.backward()
            opt.step()
        # loss_de/=len_train
        # loss_pr/=len_train
        print("validate...")
        model.eval()
        smape=0.0
        for _,data,value in loader_v:
            data=data.float().to(cuda)
            value=value.float().to(cuda)
            output=model(data)
            loss=(abs(output-value)/(abs(output)+abs(value))).mean()
            smape+=loss
        # smape/=len_validate
        status={
            "epoch":epoch,
            "model":model.state_dict()
        }
        scalars={
            'train_encode':loss_de,
            'train_predict':loss_pr,
            'train_sum':loss_pr+loss_de,
            'validate':smape
        }
        writer.add_scalars('molecule',scalars,epoch)
        writer.close()
        save(status,'checkpoint')
        epoch+=1
