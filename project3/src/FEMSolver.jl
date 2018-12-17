using LinearAlgebra

include("lagrangeInterpolation.jl")


function projectinitial(M, f⁰::Array{Float64, 1})
	M[1,:]   .= [1; zeros(Float64, size(M,1)-1)]
	M[end,:] .= [zeros(Float64, size(M,1)-1); 1]
	f⁰[1], f⁰[end] = 0.0, 0.0
	u⁰ = M \ f⁰
	return u⁰
end

function projectinitial(x, f::Function)
    u⁰ = f.(x)
    return u⁰
end

function iterate_forward(K, M, uⁿ, β)
    # M uⁿ⁺¹ = (M - β⋅K) uⁿ

    Mᵪ = copy(M) 
    # Handle u(0) = u(1) = 0 boundary conditions.

    bⁿ = (Mᵪ - β * K) * uⁿ
    bⁿ[1], bⁿ[end] = 0.0, 0.0
    Mᵪ[1,:]   .= [1; zeros(Float64, size(Mᵪ,1)-1)]
    Mᵪ[end,:] .= [zeros(Float64, size(Mᵪ,1)-1); 1]
    
    uⁿ⁺¹ = Mᵪ \ bⁿ
    return uⁿ⁺¹
end

function iterate_backward(K, M, uⁿ, β)
	# (M + β⋅K) uⁿ⁺¹ = M uⁿ
	bⁿ = M * uⁿ
	A  = M + β * K

	# Handle u(0) = u(1) = 0 boundary conditions.
    bⁿ[1], bⁿ[end] = 0.0, 0.0
    A[1,:]   .= [1; zeros(Float64, size(M,1)-1)]
    A[end,:] .= [zeros(Float64, size(M,1)-1); 1]

    uⁿ⁺¹ = A \ bⁿ
end
