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

function iterate_forward(A, B, uⁿ, β)
    # B uⁿ⁺¹ = (B - β A) uⁿ
    bⁿ   = (B - β * A) \ uⁿ
    uⁿ⁺¹ = B \ bⁿ
    return uⁿ⁺¹
end