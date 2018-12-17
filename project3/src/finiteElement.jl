using Polynomials
using UnicodePlots
using LinearAlgebra
using FastGaussQuadrature
using RowEchelon
using DelimitedFiles

include("lagrangeInterpolation.jl")
include("FEMBasis.jl")
include("FEMSolver.jl")


function f(x) 
    return sin.(π.*x)
end

function uₑ(x, t)
    return sin.(π.*x).*exp(-π^2 .* t)
end

function FEM_P2(N::Integer)
    for Nₜ = 10:5:1000
        tfile = open("../data/t$Nₜ.dat", "w")
        ufile = open("../data/u$Nₜ.dat", "w")

        nodes, elements, basis = generateP2Basis(N)

        #K, M, f⁰ = setupmatrices_gaussianquadrature(nodes, elements, basis, f)
        K, M, f⁰ = setupmatrices_analytical(nodes, elements, basis, f)
        #u⁰ = projectinitial(nodes, f)
        u⁰ = projectinitial(M, f⁰)    

        α  = 1 # Parameter in the diffusion equation
        t₀ = 0
        t₁ = 0.1 
        #Nₜ = 25
        Δt = (t₁ - t₀) / Nₜ
        β  = α * Δt
        ε = []

        uⁿ = u⁰
        t = t₀

        writedlm(tfile, t)
        writedlm(ufile, transpose(uⁿ))

        for tᵢ = 1:Nₜ
            t = t + Δt 
            #uⁿ⁺¹ = iterate_forward(K, M, uⁿ, β)
            uⁿ⁺¹ = iterate_backward(K, M, uⁿ, β)
            uⁿ = uⁿ⁺¹

            writedlm(tfile, t)
            writedlm(ufile, transpose(uⁿ))

            #if tᵢ % 1 == 0
            if tᵢ == Nₜ
                println(Δt, " ", sum(abs.(uₑ(nodes,t).-uⁿ))/N)
            end
        end
        #ΔtΔx² = Δt/(nodes[2]-nodes[1])^2
        #println("Δt/Δx² = $ΔtΔx² \n")
    end
end



FEM_P2(11)


