mutable struct Neuron{T<:Real}
    w::Vector{T}
    η::Real
    b::T
    ϕ::Function
end
Neuron(N,η,ϕ)=Neuron(rand(N),η,rand(),ϕ)

sgn(x)=x>=0 ? 1 : -1

function forward(n::Neuron{T},input::Vector{K}) where {T<:Real,K<:Real}
    (n.w'*input+n.b)|>n.ϕ
end

function update!(n::Neuron{T},input::Vector{K},d::Real) where {T<:Real,K<:Real}
    y=forward(n,input)
    e=d-y
    n.w=n.w+n.η*e*input
    n.b=n.b+n.η*e
    return e
end

data=[([x,y],x&y<1 ? -1 : 1) for x ∈ 0:1 for y ∈ 0:1]

neuron=Neuron(2,0.03,sgn)
for epoch ∈ 1:10
    for (x,y) in data
        loss=update!(neuron,x,y)
        @info loss
        println(neuron)
    end
end
