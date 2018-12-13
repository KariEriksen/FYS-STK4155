using Polynomials
using UnicodePlots
using LinearAlgebra
using FastGaussQuadrature
using RowEchelon

include("lagrangeInterpolation.jl")
include("FEMBasis.jl")
include("FEMSolver.jl")


function FEM_P2(N::Integer, M::Integer=1000)
    nodes, elements, basis = generateP2Basis(N)
    display(nodes)
    display(elements)
    display(basis)

    function transform(x, a, b)
        return x*(b-a)/2 .+ (a+b)/2
    end

    function ψ(x,i)
        return piecewise(x, basis[i])
    end

    function ∂ψ(x,i) 
        return ∂piecewise(x, basis[i])
    end
    

    #=

    function min(a::Array{<:Real,1}) 
        m = a[1]
        for i = 2:length(a)
            if a[i] < m
                m = a[i]
            end
        end
        return m
    end

    function max(a::Array{<:Real,1}) 
        m = a[1]
        for i = 2:length(a)
            if a[i] > m
                m = a[i]
            end
        end
        return m
    end

    # Integral points, weights
    x, ω = gausslegendre(M)
    #x, ω = gausschebyshev(M,1)
    #x, ω = gausschebyshev(M,2)

    A = zeros(Float64, N, N)
    B = zeros(Float64, N, N)
    f = zeros(Float64, N)

    function f(x) 
        return -6.0 .* x .+ 2.0
    end


    for i = 1:N
        bᵢ = basis[i]
        sᵢ = bᵢ.support

        a = sᵢ[1]
        b = sᵢ[end]

        uᵢf  = piecewise(x*(b-a)/2 .+ (a+b)/2, bᵢ) .* f(x*(b-a)/2 .+ (a+b)/2)
        ∫uᵢf = (b-a)/2 * sum(uᵢf .* ω)
        B[i] = ∫uᵢf 

        for j = i:N
            bⱼ = basis[j]
            sⱼ = bⱼ.support

            if sᵢ[end] < sⱼ[1]
                A[i,j] = 0.0
                A[j,i] = 0.0
            end

            a = min(sᵢ)
            b = max(sⱼ)
            ∂uᵢ∂uⱼ = ∂piecewise(x*(b-a)/2 .+ (a+b)/2, bᵢ) .* 
                     ∂piecewise(x*(b-a)/2 .+ (a+b)/2, bⱼ)
            ∫∂uᵢ∂uⱼ = (b-a)/2 * sum(∂uᵢ∂uⱼ .* ω)

            # Minus sign from the integration by parts in Galerkin method
            A[i,j] = -∫∂uᵢ∂uⱼ
            A[j,i] = -∫∂uᵢ∂uⱼ
        end
    end
    A[1,:]    .= 0.0
    A[end,:]  .= 0.0
    A[1,1]     = 1.0
    A[end,end] = 1.0
    B[1]   = 0.0
    B[end] = 0.0

    function showarray(array::Array{<:Number,<:Number})
        display(array)
        println("")
    end


    uₐₚₚ = A \ B
    #N = 100
    x = nodes #collect(range(0.0, stop=1.0, length=N))
    uₑₓ = -x.^3 .+ x.^2 #x - x.^2

    function abs!(array::Array{<:Number, 1})
        for i = 1:length(array)
            array[i] = abs(array[i])
        end
        return array
    end

    #println(lineplot(nodes, uₐₚₚ))
    #println(lineplot(x, uₑₓ))
    println("N: ", N, "  e: ", sum(abs!(uₑₓ[2:end-1] .- uₐₚₚ[2:end-1]))/(N-2))
    =#
end

FEM_P2(5)

for N = 5:10:101
end


