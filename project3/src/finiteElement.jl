using Polynomials
using UnicodePlots
using LinearAlgebra
using FastGaussQuadrature
using RowEchelon

include("lagrangeInterpolation.jl")
include("FEMBasis.jl")
include("FEMSolver.jl")


function f(x) 
    return sin.(π.*x)
end


function FEM_P2(N::Integer, M::Integer=1000)
    nodes, elements, basis = generateP2Basis(N)

    A, B, b, f⁰ = setupMatrices(nodes, elements, basis, f)

    u⁰ = projectinitial(nodes, f)
    
    println(lineplot(nodes, u⁰))

    α  = 1    # Parameter in the diffusion equation
    Nₜ = 100
    Δt = 0.01 
    β  = α * Δt

    uⁿ = u⁰
    for tᵢ = 1:1
        uⁿ⁺¹ = iterate_forward(A, B, uⁿ, β)
        uⁿ = uⁿ⁺¹

        if true #tᵢ % 10 == 0
            println(lineplot(nodes, uⁿ))
        end
    end
end



FEM_P2(5)


