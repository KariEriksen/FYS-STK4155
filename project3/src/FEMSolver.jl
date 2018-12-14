using LinearAlgebra

include("lagrangeInterpolation.jl")


function projectinitial(B, f⁰::Array{Float64, 1})
	u⁰ = B \ f⁰
	return u⁰
end

function projectinitial(x, f::Function)
    u⁰ = f.(x)
    return u⁰
end

function iterate_forward(K, M, uⁿ, β)
    # M uⁿ⁺¹ = (M - β⋅K) uⁿ
    bⁿ   = (M - β * K) * uⁿ
    uⁿ⁺¹ = M \ bⁿ
    return uⁿ⁺¹
end