using Polynomials
using FastGaussQuadrature
include("lagrangeInterpolation.jl")

function piecewise(xx    ::Array{<:Real,1}, 
                   pieces::Array{<:Real,1}, 
                   lower ::Poly{Float64}, 
                   upper ::Poly{Float64})
    y = copy(xx)
    for i = 1:length(xx)
        x = xx[i]
        if x < pieces[1]
            y[i] = 0
        elseif x >= pieces[1] && x < pieces[2]
            y[i] = lower(x)
        elseif x >= pieces[2] && x <= pieces[3]
            y[i] = upper(x)
        else 
            y[i] = 0
        end
    end

    if abs(pieces[end] - 1.0) < 1e-10
        if abs(lower(1.0) - 1.0) < 1e-10
            y[end] = 1.0
        end
    end
    
    return y
end

struct basisPolynomial
    basisPolynomial() = new([0.0], 0, 0, 0, 0)
    basisPolynomial(s    ::Array{<:Real, 1}, 
                    p₀   ::Poly{<:Real}, 
                    ∂p₀∂x::Poly{<:Real}, 
                    p₁   ::Poly{<:Real}, 
                    ∂p₁∂x::Poly{<:Real},
                    c::Complex) = new(s, p₀, ∂p₀∂x, p₁, ∂p₁∂x)

    support::Array{<:Real, 1}
    p₀     ::Poly{<:Real}
    ∂p₀∂x  ::Poly{<:Real}
    p₁     ::Poly{<:Real}
    ∂p₁∂x  ::Poly{<:Real}
end

function piecewise(x::Array{<:Real,1}, basis::basisPolynomial)
    return piecewise(x, 
                     basis.support, 
                     basis.p₀,
                     basis.p₁)
end

function ∂piecewise(x::Array{<:Real,1}, basis::basisPolynomial)
    return piecewise(x, 
                     basis.support, 
                     basis.∂p₀∂x,
                     basis.∂p₁∂x)
end

function generateP2Basis(N::Integer) 
    # In order to use P2 basis, we need the total number of points to be odd.
    @assert N >= 5
    @assert N%2 == 1

    nodes    = collect(range(0.0, stop=1.0, length=N))
    elements = [[j for j = i:i+2] for i = 1:2:N-2]

    # Generate basis of P2 functions
    basis = []

    for i = 1:length(elements)
        for j = 1:length(elements[i])
            p₀ = Poly([0.0])
            p₁ = Poly([0.0])
            ∂p₀∂x = Poly([0.0])
            ∂p₁∂x = Poly([0.0])

            if j == 1
                if i == 1
                    p₀    = Poly([0.0])
                    ∂p₀∂x = Poly([0.0])

                    ind = elements[1]
                    support = [nodes[ind[1]], nodes[ind[1]], nodes[ind[end]]]
                else 
                    n     = [nodes[elements[i-1][1]], nodes[elements[i-1][2]], nodes[elements[i-1][end]]]
                    p₀    = LagrangeInterpolatingPolynomial(n, length(elements[i-1]))
                    ∂p₀∂x = polyder(p₀)   
                end
                n     = [nodes[elements[i][1]], nodes[elements[i][2]], nodes[elements[i][end]]]
                p₁    = LagrangeInterpolatingPolynomial(n, 1)
                ∂p₁∂x = polyder(p₁)

                if i != 1 
                    ind0 = elements[i-1]
                    ind1 = elements[i]
                    support = [nodes[ind0[1]], nodes[ind1[1]], nodes[ind1[end]]]
                end
            
            elseif j == length(elements[i])
                if i == length(elements)
                    p₁      = Poly([0.0])
                    ∂p₁∂x   = Poly([0.0]) 

                    ind     = elements[end]
                    support = [nodes[ind[1]], nodes[ind[end]], nodes[ind[end]]]
                else 
                    n     = [nodes[elements[i+1][1]], nodes[elements[i+1][2]], nodes[elements[i+1][end]]]
                    p₁    = LagrangeInterpolatingPolynomial(n, 1)
                    ∂p₁∂x = polyder(p₁)
                end
                n     = [nodes[elements[i][1]], nodes[elements[i][2]], nodes[elements[i][end]]]
                p₀    = LagrangeInterpolatingPolynomial(n, length(elements[i]))
                ∂p₀∂x = polyder(p₀) 

                if i != length(elements)
                    ind0 = elements[i]
                    ind1 = elements[i+1]
                    support = [nodes[ind0[1]], nodes[ind1[1]], nodes[ind1[end]]]
                end
            
            else
                n     = [nodes[elements[i][1]], nodes[elements[i][2]], nodes[elements[i][end]]]
                p₀    = LagrangeInterpolatingPolynomial(n, j)
                ∂p₀∂x = polyder(p₀) 

                p₁      = p₀
                ∂p₁∂x   = ∂p₀∂x
                support = [nodes[elements[i][1]], nodes[elements[i][2]], nodes[elements[i][end]]]
            end

            if ! (i != 1 && j == 1)
                push!(basis, basisPolynomial(support, p₀, ∂p₀∂x, p₁, ∂p₁∂x, 1im))
            end
        end
    end
    return nodes, elements, basis
end

function transform(x, a, b)
    return x*(b-a)/2 .+ (a+b)/2
end

function ψ(x, bᵢ)
    return piecewise(x, bᵢ)
end

function ψₓ(x, bᵢ) 
    return ∂piecewise(x, bᵢ)
end

function setupMatrices(nodes, elements, basis, initialcondition)
    N  = length(nodes)
    K  = zeros(Float64, N, N)
    M  = zeros(Float64, N, N)
    f⁰ = zeros(Float64, N)

    # Integral points, weights
    x, ω = gausslegendre(Int(1e5))

    for i = 1:N
        bᵢ = basis[i]
        sᵢ = bᵢ.support

        a = sᵢ[1]
        b = sᵢ[end]
        χ = transform(x, a, b)

        ∑ωψⁱf = sum(ψ(χ, bᵢ) .* f.(χ) .* ω)
        ∫ωψⁱf = (b-a) / 4 * ∑ωψⁱf #(b-a) / 2 * ∑ωψⁱf

        f⁰[i] = ∫ωψⁱf

        for j = i:N
            # Since j ≥ i, the support of ψʲ, [aʲ, bʲ], is guaranteed to be larger than
            # or equal to the support for ψⁱ, [aⁱ, bⁱ]. I.e. aʲ ≥ aⁱ, bⁱ ≥ bʲ.
            bⱼ = basis[j]
            sⱼ = bⱼ.support

            # If there is no overlap between the supports of ψⁱ and ψʲ, the Integral
            # vanishes.
            if ! (sᵢ[end] < sⱼ[1])

                a = minimum(sᵢ)
                b = maximum(sⱼ)
                χ = transform(x, a, b)
                
                ∑ωψⁱₓψₓʲ = sum(ψₓ(χ, bᵢ) .* ψₓ(χ, bⱼ) .* ω)
                ∑ωψⁱψʲ   = sum( ψ(χ, bᵢ) .*  ψ(χ, bⱼ) .* ω)

                ∫ωψⁱₓψₓʲ = (b-a) / 4 * ∑ωψⁱₓψₓʲ #(b-a) / 2 * ∑ωψⁱₓψₓʲ
                ∫ωψⁱψʲ   = (b-a) / 4 * ∑ωψⁱψʲ   #(b-a) / 2 * ∑ωψⁱψʲ

                K[i,j] = ∫ωψⁱₓψₓʲ
                K[j,i] = ∫ωψⁱₓψₓʲ

                M[i,j] = ∫ωψⁱψʲ
                M[j,i] = ∫ωψⁱψʲ
            end
        end
    end
    return K, M, f⁰
end

