using Polynomials
using UnicodePlots

function LagrangeInterpolatingPolynomial(x::Array{<:Real,1}, j::Int)
    p = Poly([1.0])
    d = 1.0
    for k = 1:length(x)
        if k != j 
            pₖ = Poly([-x[k], 1.0])
            dₖ = (x[j] - x[k])
            p *= pₖ
            d *= dₖ
        end
    end
    return p / d
end

function ∂LagrangeInterpolatingPolynomial(x::Array{<:Real,1}, j::Int)
    p = LagrangeInterpolatingPolynomial(x, j)
    ∂p∂x = polyder(p, 1)
    return ∂p∂x
end

function piecewise(x::Number, pieces::Array{<:Real,1}, lower::Function, upper::Function)
    if x < pieces[1]
        return 0
    elseif x > pieces[1] && x < pieces[2]
        return lower(x)
    elseif x >= pieces[2] && x < pieces[3]
        return upper(x)
    else 
        return 0
    end
end

struct basisPolynomial
    basisPolynomial() = new([0.0], Poly([0.0]), Poly([0.0]), Poly([0.0]), Poly([0.0]))
    basisPolynomial(s::Array{<:Real, 1}, 
                    p₀::Poly{<:Real}, 
                    ∂p₀∂x::Poly{<:Real}, 
                    p₁::Poly{<:Real}, 
                    ∂p₁∂x::Poly{<:Real}) = new(s, p₀, ∂p₀∂x, p₁, ∂p₁∂x)

    support::Array{<:Real, 1}
    p₀   ::Poly{<:Real}
    ∂p₀∂x::Poly{<:Real}
    p₁   ::Poly{<:Real}
    ∂p₁∂x::Poly{<:Real}
end

# In order to use P2 basis, we need the total number of points to be odd.
N = 7
@assert N >= 5
@assert N%2 == 1

nodes    = collect(range(0.0, stop=1.0, length=N))
elements = [[float(j) for j = i:i+2] for i = 1:2:N-2]

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
                p₀      = Poly([0.0])
                ∂p₀∂x   = Poly([0.0])
                support = elements[1]  
            else 
                p₀      = LagrangeInterpolatingPolynomial(elements[i-1], length(elements[i-1]))
                ∂p₀∂x   = polyder(p₀)   

                p₁      = LagrangeInterpolatingPolynomial(elements[i], 1)
                ∂p₁∂x   = polyder(p₁)
                support = deleteat!(copy(elements[i-1]), length(elements[i-1]))
                support = vcat(support, elements[i])
            end

        elseif j == length(elements[i])
            if i == length(elements)
                p₁      = Poly([0.0])
                ∂p₁∂x   = Poly([0.0]) 
                support = elements[length(elements)]
            else 
                p₀      = LagrangeInterpolatingPolynomial(elements[i], length(elements[i]))
                ∂p₀∂x   = polyder(p₀) 

                p₁      = LagrangeInterpolatingPolynomial(elements[i+1], 1)
                ∂p₁∂x   = polyder(p₁)
                support = deleteat!(copy(elements[i]), length(elements[i]))
                support = vcat(support, elements[i+1])
            end
        
        else
            p₀      = LagrangeInterpolatingPolynomial(elements[i], j)
            ∂p₀∂x   = polyder(p₀) 

            p₁      = p₀
            ∂p₁∂x   = ∂p₀∂x
            support = elements[i]
        end

        push!(basis, basisPolynomial(support, p₀, ∂p₀∂x, p₁, ∂p₁∂x))
    end
end








