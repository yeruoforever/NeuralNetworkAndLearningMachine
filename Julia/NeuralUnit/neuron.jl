using Random 
using Plots
using Zygote

mutable struct Neuron{T <: Real}
    w::Vector{T}    # 权重
    η::Real         # 学习率
    b::T            # 偏置
    ϕ::Function     # 激活函数
end

Neuron(N,η,ϕ) = Neuron(rand(N), η, rand(), ϕ)

function forward(n::Neuron{T}, input::Vector{K}) where {T <: Real,K <: Real}
    σ = (n.w' * input + n.b) 
    σ |> n.ϕ, gradient(n.ϕ,σ)[1]
end

function update!(n::Neuron{T}, input::Vector{K}, d::Real) where {T <: Real,K <: Real}
    y, ∂y = forward(n, input)
    if isnothing(∂y)
        ∂y=one(y)
    end
    e = d - y
    n.w = n.w + n.η * e * input .* ∂y
    n.b = n.b + n.η * e * ∂y
    return e
end


# 生成训练集
data = [([x,y], x & y==1 ? 1 : -1) for x ∈ 0:1 for y ∈ 0:1]
# 定义神经元
sgn(x)=x>=0 ? 1 : -1
neuron = Neuron(2, 0.3, sgn)
# 训练
for epoch ∈ 1:100
    for (x, y) in data
        loss = update!(neuron, x, y)
        @info loss^2
        println(neuron)
    end
end


# 拟合y=2*x^2+1
using Plots
using Random
# 生成训练集
X = -3:0.01:3 |> collect
f(x,μ,σ) = 2x^2 + 1 + randn() * σ + μ
f(x) = f(x, 0, 0.3)
Y = f.(X)
# 定义神经元
func = Neuron(2, 0.001, identity)
h(x) = func.w' * [x,x^2] + func.b
# 训练
for (x1, x2, y) in zip(X, X.^2, Y)
    loss = update!(func, [x1,x2], y)
    @info loss^2
end
# 拟合效果绘图
plot(X,Y,label = "real")
plot!(X,h,label = "predict")