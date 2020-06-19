using Flux
using Flux:@epochs,onehotbatch,onecold,gradient
using Flux.Data:DataLoader
using Flux.Data.MNIST:images,labels

@info "建立模型"
leNet5=Flux.Chain(
    x->reshape(cat(x...,dims=3),28,28,1,:),
    Conv((5,5),1=>6,stride=1),
    DepthwiseConv((2,2),6=>6,sigmoid,stride=(2,2)),
    Conv((5,5),6=>16,stride=1),
    DepthwiseConv((2,2),16=>16,sigmoid,stride=(2,2)),
    Conv((4,4),16=>120),
    x->reshape(x,120,:),
    Dense(120,84,sigmoid),
    Dense(84,10),
    softmax
)

@info "生成数据集"
X_train=Array{Float32,2}.(images())
Y_train=onehotbatch(labels(),0:9) 
train_loader=DataLoader(X_train,Y_train,batchsize=128)

@info "模型配置"
opt=Descent(0.3)
ps=params(leNet5)
leNet5(X_train[1])

@info "开始训练"
for epoch in 1:10
    @info string("\t第",epoch,"回合")
    for (x,y) in train_loader
        gs=gradient(ps) do 
            ŷ=leNet5(x)
            loss=Flux.crossentropy(ŷ,y)
        end
        Flux.Optimise.update!(opt,ps,gs)
    end
end
@info "训练结束"


X_test=Array{Float32,2}.(images(:test))
Y_test=onehotbatch(labels(:test),0:9)
test_loader=DataLoader(X_test,Y_test,batchsize=128)
@info "计算测试集正确率"
function accuracy(model,loader)
    num=0
    correct=0
    for (x,y) in loader
        ŷ=model(x)
        correct+=sum(onecold(ŷ).==onecold(y))
        num+=size(y)[end]
    end
    acc=100.0*correct/num
end
acc=accuracy(leNet5,test_loader)
println("accuracy:$(acc)%.")