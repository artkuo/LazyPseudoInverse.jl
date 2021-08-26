# LazyPseudoinverse
Experimental package that evaluates pseudoinverse `inv(A'A)A'b` lazily, as if `A\b`. The naive user may write out
the pseudoinverse fully, unaware of `\`, which is faster and more accurate. (Even for square matrices, one should rarely do `inv(A)*b`, in favor of `A\b`.) This package acts lazily on the operations leading up to a pseudoinverse, and will perform a (better) QR factorization even if the user never knew about `\`. 

```julia
julia> A = [1 2 3; 4 5 6; 7. 8 10; 11 13 19];
julia> b = [5, 8, 10, 11.];
julia> inv(A'A)A'b # looks like normal matrix operation
3-element Vector{Float64}:
 -4.679738562091531
  8.222222222222245
 -2.3333333333333317

 julia> inv(A'A)A'b == A\b # except really a QR solve!
 true
```
On the way toward pseudoinverse, other operations are recognized, and the package tries to do the numerically
best operation even if there is no pseudoinverse at the end.
For example, operations like `A'A`
are recognized as lazy positive definite, so that `inv(A'A)` 
can use a Cholesky decomposition (instead of the default LU). If the pseudoinverse `inv(A'A)A'` is materialized, it
will return the `pinv`, superior to performing the operations literally.

Warning: This is but a proof of concept, and lacks many 
methods that need to be defined to make this work generally.
Just enough calculations work to demonstrate the principles.

There are a few steps to attaining this extreme laziness.

## 1. Lazy matrix inner product `A'A`
Actually stored as `MatrixInnerProd`, which can be displayed but in reality is lazy:
```julia
julia> A'A
3×3 MatrixInnerProd{Float64, Matrix{Float64}}:
 187.0  221.0  306.0
 221.0  262.0  363.0
 306.0  363.0  506.0    
```
It's actually a container for `A` awaiting more operations
that can take advantage of its positive definiteness,
such as the inverse:

## 2.  Lazy inverse matrix inner product `inv(A'A)`
```julia
julia> inv(A'A)
3×3 InvMatrixInnerProd{Float64, Matrix{Float64}}:
  1.74946   -1.62963    0.111111
 -1.62963    2.14815   -0.555556
  0.111111  -0.555556   0.333333  
```
Again only a container, but if materialized it actually
does a cholesky decomposition of A'A instead of LU.
```julia
julia> At = collect(A') # materialized adjoint
julia> collect(inv(A'A)) == inv(cholesky(At*A)) 
true
```    
If not materialized, it's lazily awaiting
yet another operation, such as multiplication by `A'`:

## 3.  Lazy left pseudoinverse `inv(A'A)A'`
This is yet another lazy operation, awaiting multiplication
against something like `b`. If materialized, it will produce
a `pinv` pseudoinverse, which is again superior to
literally computing the quantity by traditional means. 
```julia
julia> inv(A'A)A'
3×4 LazyLeftPseudoinverse{Float64, Matrix{Float64}}:
 -1.17647      -0.48366    0.320261   0.169935
  1.0           0.888889   0.222222  -0.555556
  1.22242e-16  -0.333333  -0.333333   0.333333  

julia> collect(inv(A'A)A') == pinv(A)
true
```
## 4.  Finally a pseudoinverse `inv(A'A)A'b`
If the pseudoinverse actually comes to fruition, none of the 
above operations are performed. Instead, the QR-based `\` is
substituted:
```julia
julia> inv(A'A)A'b
3-element Vector{Float64}:
 -4.679738562091531
  8.222222222222245
 -2.3333333333333317

julia> inv(A'A)A'b == A\b
true
 ```
 ## A few other things also work
 For example, even if the code doesn't complete a pseudoinverse, the package will still take advantage of positive definite `A'A`, and use the preferred
 Cholesky decomposition.
```julia
julia> B*inv(A'A)
4×3 Matrix{Float64}:
 -1.17647    1.0      -6.38378e-15
 -0.559913   5.74074  -3.77778
  4.30283   -3.92593   0.222222
 -5.0305     6.74074  -1.77778

julia> B*inv(A'A) == B*inv(cholesky(At*A)) # left-multiply
true

julia> inv(A'A)*B' == cholesky(At*A)\B' # right-multiply
true
```
## What's the point?
There is no point. This is an exercise to see far lazy matrix evaluations can go, and nothing more. All users should really learn to use `\` and nobody should type `inv(A'A)A'b` into Julia. But some users don't know much about numerical analysis, and this package demonstrates that it's possible for
the language to make up for some deficiencies.

Anyone curious about how it's done should take a look at the code. Almost everything is a simple matter of some custom struct types and multiple dispatch. There is a bit of type piracy involved, to hijack `A'A` and make it lazy. Also, note that despite the principled laziness, there are a lot of wasted computations when anything is displayed for output, which
causes things like `inv(A'A)` in REPL to be materialized for cosmetic purposes. 

Note that only left pseudoinverses are performed at this time. Right pseudoinverses are readily attainable, but not implemented because there's no point.
## References
There are prior examples of packages that implement laziness, often in much better ways than here.
*  [LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl) enables lazy operations for high performance
*  [InfiniteArrays.jl](https://github.com/JuliaArrays/InfiniteArrays.jl) is work in progress toward (lazy) infinite-dimensional computations
*  [PDMats.jl](https://github.com/JuliaStats/PDMats.jl) is a much more serious implementation of positive definite matrices. It does some operations eagerly, for example storing the Cholesky decomposition during struct contruction. 
