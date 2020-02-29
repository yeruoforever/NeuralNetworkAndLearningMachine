from argparse import ArgumentParser
from gc import collect
from pandas import DataFrame, read_csv,merge,Series
from torch import load, save,randn,device,no_grad,Tensor
from torch.nn import Linear, Module, MSELoss, Parameter, Sequential, Tanh
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class MyModel(Module):
    def __init__(self,in_features:int,out_features:int):
        super(MyModel,self).__init__()
        # self.mean_in=Parameter(randn(1,in_features))
        # self.deta_in=Parameter(randn(1,in_features))
        # self.mean_out=Parameter(randn(1,out_features))
        # self.deta_out=Parameter(randn(1,out_features))
        self.encode=Sequential(
            Linear(in_features,2048),
            Tanh(),
            Linear(2048,1024),
            Tanh(),
            Linear(1024,256),
            Tanh()
        )
        self.features=Sequential(
            Linear(256,1024),
            Tanh(),
            Linear(1024,2048),
            Tanh(),
            Linear(2048,2048),
            Tanh(),
            Linear(2048,256),
            Tanh(),
            Linear(256,6)
        )
    def forward_en(self,input):
        # input=(input-self.mean_in)/self.deta_in**2
        x=self.encode(input)
        return x

    def forward_feature(self,input):
        output=self.features(input)
        # output=output*self.deta_out**2+self.mean_out
        return output

    def forward(self,input):
        x=self.forward_en(input)
        out=self.forward_feature(x)
        return out

def analysis(data:DataFrame):
    indexs=data['id'].values.astype(str)
    data=data.drop('id',axis=1)
    columns_all=list(data.columns)
    columns_targets=['p1','p2','p3','p4','p5','p6']
    columns_features=[column for column in columns_all if column not in columns_targets]
    features=data[columns_features].values.astype(float)
    targets=data[columns_targets].values.astype(float)
    return indexs,features,targets

class MyDatasets(Dataset):
    def __init__(self,indexs,features,targets):
        super(MyDatasets,self).__init__()
        self.indexs=indexs
        self.features=features
        self.targets=targets

    def __getitem__(self,index):
        names=self.indexs[index]
        features=self.features[index]
        targets=self.targets[index]
        return names,features,targets

    def __len__(self):
        return len(self.indexs)

def partition():
    features=read_csv("candidate_train.csv")
    targets=read_csv("train_answer.csv")
    data_all=merge(features,targets,on='id')
    cdt=data_all.std()>0
    cdt=cdt[cdt]
    columns=Series([True],index=['id'])
    columns=columns.append(cdt)
    data_all=data_all[columns.index]
    data=data_all.drop('id',axis=1)
    data=(data-data.mean())/data.std()
    data['id']=data_all['id']
    data_train=data.sample(frac=0.7,axis=0)
    data_validate=data[~data.index.isin(data_train.index)]
    train=MyDatasets(*analysis(data_train))
    validate=MyDatasets(*analysis(data_validate))
    save(validate,'validate')
    save(train,'train')
    config={
        'mean':data_all.mean(),
        'std':data_all.std(),
        'columns':columns.index
    }
    save(config,'mean_std')


parser=ArgumentParser(description="模型训练程序")
parser.add_argument('-r','--resume',help="从上次的断点继续训练",action='store_true')
parser.add_argument('-g','--gpu',help="使用GPU训练和测试，默认使用cuda:0",default=0,type=int)
parser.add_argument('-p','--partition',help="划分训练集、验证集",action='store_true')
parser.add_argument('-s','--save',help="存储测试集结果",action='store_true')
args=parser.parse_args()


if args.partition:
    print("Geting partitions of train and validate...")
    partition()
    collect()


validate=load('validate')
train=load('train')
loader_train=DataLoader(train,batch_size=16)
loader_validate=DataLoader(validate,batch_size=16)


epoch=0
epoch_max=400

writer=SummaryWriter(log_dir='./run')

cuda=device('cpu')
if args.gpu==0:
    cuda=device('cuda:0')
    print('Using gpu:0')
else:
    cuda=device('cuda:1')
    print('Using gpu:1')

model=MyModel(3126,6).to(cuda)

if args.resume:
    try:
        checkpoint=load('checkpoint')
        epoch=checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model'])
        print("Training from epoch %d"%(epoch))
    except Exception as e:
        print(e)
        print("Failed recume, training from epoch 1.")

opt=SGD(model.parameters(),lr=0.0001)
criterion=MSELoss()

while epoch<epoch_max:
    model.train()
    print("Training...")
    loss_train=0.
    for _,features,targets in loader_train:
        inputs=features.float().to(cuda)
        targets=targets.float().to(cuda)
        outputs=model(inputs)
        loss=criterion(outputs,targets)
        loss_train+=loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    print("Validating...")
    loss_validate=0.
    with no_grad():     
        for _,features,targets in loader_validate:
            inputs=features.float().to(cuda)
            targets=targets.float().to(cuda)
            outputs=model(inputs)
            loss=criterion(outputs,targets)
            loss_validate+=loss.item()
    print("Saving and ploting...")
    state={
        'model':model.state_dict(),
        'epoch':epoch
    }
    save(state,'checkpoint')
    scalars={
            'train':loss_train/0.7,
            'validate':loss_validate/0.3
    }
    writer.add_scalars('molecule',scalars,epoch)
    writer.close()
    epoch+=1

collect()
if args.save:
    print('Saving...')
    test=read_csv("candidate_val.csv")
    config=load('mean_std')
    std=config['std']
    mean=config['mean']
    test_names=test['id']
    data=test.drop('id',axis=1)
    columns_all=config['columns']
    columns_targets=['p1','p2','p3','p4','p5','p6']
    columns_features=[column for column in columns_all if column not in columns_targets]
    columns_features.remove('id')
    features=data[columns_features].values.astype(float)
    model=model.cpu()
    std_features=std[columns_features].values.astype(float)
    std_targets=std[columns_targets].values.astype(float)
    mean_features=mean[columns_features].values.astype(float)
    mean_targets=mean[columns_targets].values.astype(float)
    features=(features-mean_features)/std_features
    collect()
    with no_grad():
        outputs=model(Tensor((features)))
    targets=outputs*std_targets+mean_targets
    df=DataFrame(
        targets.numpy(),
        index=test_names,
        columns=columns_targets
    )
    df.to_csv('val_answer.csv')    

