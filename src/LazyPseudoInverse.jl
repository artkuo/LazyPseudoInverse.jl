"""Experimental package that evaluates pseudoinverses such as `inv(A'A)A'b` lazily, as if `A\b`.
Note that many normal matrix operations will fail, because only a few sample methods have been
defined.
"""
module LazyPseudoinverse

using LinearAlgebra, PDMats
import Base: *, \, promote_op, inv, collect, Matrix, size # overloads
import LinearAlgebra: matprod, cholesky
import PDMats.PDMat

export MatrixInnerProd, adjointmul, InvMatrixInnerProd, LazyLeftPseudoinverse

abstract type AbstractMatrixInnerProd{T} <: AbstractMatrix{T} end 
abstract type AbstractInvMatrixInnerProd{T} <: AbstractMatrix{T} end

"""
`MatrixInnerProd(A::AbstractMatrix)`

Construct a matrix inner product `A'A` from `A`. This stores `A` as a field and
can act like a positive definite matrix. For example, `inv` and `\` are defined,
although `inv` constructs a lazy InvMatrixInnerProd.
Use `collect` to materialize an actual matrix. It can also be converted to 
`PDMat(MatrixInnerProd(A))` if `using PDMats`.
"""
struct MatrixInnerProd{T,S<:AbstractMatrix{T}} <: AbstractMatrixInnerProd{T}
    A::S
end

Base.collect(AtA::MatrixInnerProd) = adjointmul(AtA.A',AtA.A)
Base.Matrix(AtA::MatrixInnerProd) = collect(AtA)
Base.size(AtA::MatrixInnerProd) = (size(AtA.A,2),size(AtA.A,2))
function cholesky(AtA::MatrixInnerProd)
    c = cholesky(collect(AtA), check = false)
    if !issuccess(c)  lu(collect(AtA)); end
    return c
end

# When user shows MatrixInnerProd, print the result (A'A) as if normal, even though it's really lazy
Base.print_array(io::IO, A::MatrixInnerProd) = Base.print_array(io, adjointmul(A.A',A.A))
\(AtA::MatrixInnerProd, B::AbstractVector) = cholesky(AtA) \ B 
\(AtA::MatrixInnerProd, B::AbstractMatrix) = cholesky(AtA) \ B 
*(AtA::MatrixInnerProd, B::AbstractVector) = collect(AtA)*B 
*(AtA::MatrixInnerProd, B::AbstractMatrix) = collect(AtA)*B 
# Only a few methods defined here, so many matrix operations will fail, e.g. A'A+B


"""
`adjointmul(A', B) performs inner product of `A` and `B`, `A'B`. This is originally `*`
from `matmul.jl` (which has been type-pirated for LazyPseudoinverse.jl). To actually
complete `A'*B`, use `adjointmul(A', B)`
"""
function adjointmul(A::Adjoint,B::AbstractMatrix) # A'*B
    TS = promote_op(matprod, eltype(A), eltype(B)) # this is what * normally does in matmul.jl:151
    mul!(similar(B, TS, (size(A,1), size(B,2))), A, B)
end

"""
`A'*B` is overloaded to perform a normal matrix multiplication, except when fed `A'A` which
returns a `MatrixInnerProd` which stores `A` and acts like positive definite matrix.
Warning: This is type piracy of `Base.*`, be careful!
"""
function *(A::Adjoint,B::AbstractMatrix) # A'*B
    if A.parent === B # might be a bit looser to do ==
        MatrixInnerProd(A.parent)
    else
        adjointmul(A, B) # do a regular matmul
    end
end
PDMats.PDMat(A::MatrixInnerProd) = PDMat(adjointmul(A.A',A.A)) # sample conversion to positive definite matrix

"""
`InvMatrixInnerProd(A::AbstractMatrix)`

Construct an inverse matrix inner product `inv(A'A`) from `A`. This stores `A` as a field and
can act like an inverted positive definite matrix. 
Use `collect` to materialize an actual matrix. It can also be converted to 
`PDMat(MatrixInnerProd(A))` when `using PDMats`.
"""
struct InvMatrixInnerProd{T,S<:AbstractMatrix{T}} <: AbstractInvMatrixInnerProd{T}
    A::S
end
Base.collect(invAtA::InvMatrixInnerProd) = inv(cholesky(invAtA.A'invAtA.A))
Base.Matrix(invAtA::InvMatrixInnerProd) = collect(invAtA)
Base.size(invAtA::InvMatrixInnerProd) = (size(invAtA.A,2),size(invAtA.A,2))
# When user shows InvMatrixInnerProd, print the result inv(A'A) as computed regularly:
Base.print_array(io::IO, A::InvMatrixInnerProd) = Base.print_array(io, inv(cholesky(A.A'A.A)))
PDMats.PDMat(A::InvMatrixInnerProd) = PDMat(inv(adjointmul(A.A',A.A))) # Cholesky decomposition

inv(AtA::MatrixInnerProd) = InvMatrixInnerProd(AtA.A) # lazy inverse

function *(invAtA::InvMatrixInnerProd,B::Adjoint) # inv(A'A)*B'
    if invAtA.A === B.parent  # B = A, so inv(A'A)*A'
        LazyLeftPseudoinverse(invAtA.A) # return the pseudo-inverse, ready to be right-multiplied
    else
        cholesky(invAtA.A'invAtA.A)\B # use linear solve instead of explicit inverse
    end
end

# Ensure that stuff like inv(A'A)B, or B*inv(A'A) still works as normal, although to get
# useful functionality many other methods need to be defined
*(invAtA::InvMatrixInnerProd,B::AbstractVector) = cholesky(invAtA.A'invAtA.A)\B 
*(invAtA::InvMatrixInnerProd,B::AbstractMatrix) = cholesky(invAtA.A'invAtA.A)\B
*(B::AbstractMatrix,invAtA::InvMatrixInnerProd) = B*collect(invAtA)


"""
`LazyLeftPseudoinverse(A::AbstractMatrix)`

Construct equivalent of `inv(A'A)A'` from `A`. This stores `A` as a field and
can act (and displays) like a pseudo-inverse `pinv(A)`. But when actually multiplied
against `b`, it computes `A\\b` using QR. 
Use `collect` to materialize an actual matrix. 
"""
struct LazyLeftPseudoinverse{T,S<:AbstractMatrix{T}} <: AbstractMatrix{T}
    A::S
end

Base.size(pending::LazyLeftPseudoinverse) = reverse(size(pending.A))
Base.print_array(io::IO, pending::LazyLeftPseudoinverse) = Base.print_array(io, pinv(pending.A))
*(pending::LazyLeftPseudoinverse, B::AbstractVector) = pending.A \ B
*(pending::LazyLeftPseudoinverse, B::AbstractMatrix) = pending.A \ B
*(B::AbstractMatrix,pending::LazyLeftPseudoinverse) = B*pinv(pending.A)
Base.collect(lazyLeftpinv::LazyLeftPseudoinverse) = pinv(lazyLeftpinv.A)
Base.Matrix(lazyLeftpinv::LazyLeftPseudoinverse) = collect(lazyLeftpinv)

end # module
