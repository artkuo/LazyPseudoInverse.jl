using LazyPseudoinverse, LinearAlgebra
using Test

@testset "LazyPseudoInverse.jl" begin
    A = [1 2 3; 4 5 6; 7. 8 10; 11 13 19]            # a few handy matrices/vectors for testing
    At = collect(A') # the unadulterated adjoint
    B = [1 2 3; 8 9 1; 5. 3 4; 5 9 8]
    b = [5, 8, 10, 11.]
    c = [1.,2.,3.]
    @test typeof(A'A) <: MatrixInnerProd
    @test collect(A'A) == At*A                       # collecting materializes the actual matrix
    @test A'A*collect(B') == At*A*B'                 # right-multiply should just complete
    @test adjointmul(A',A) == At*A                   # same thing, except using adjointmul
    @test A'B == At*B                                # non-inner product 
    @test inv(A'A)*B' == cholesky(At*A)\B'           # lazy inverse works with adjoint B'
    @test inv(A'A)c  == cholesky(At*A)\c             # lazy inverse works with vector
    @test inv(A'A)*collect(B') == cholesky(At*A)\B'  # works with regular right-multiplied matrix
    @test B*inv(A'A) ≈ B*inv(cholesky(At*A))         # left-multiplied matrix with lazy PD inverse
    @test inv(A'A)A'b == A\b                         # acts like \ with vector b, using QR
    @test inv(A'A)A'B == A\B                         # acts like \ with matrix B
    @test inv(A'A)A'B == qr(A,Val(true))\B           # ...which should use QR

# Also manually check that show works:  should say MatrixInnerProd
# and show the materialized output.
#  show(stdout, MIME("text/plain"), A'A)           -> MatrixInnerProd
#  show(stdout, MIME("text/plain"), inv(A'A))      -> InvMatrixInnerProd
#  show(stdout, MIME("text/plain"), inv(A'A)A')    -> LazyLeftPseudoinverse
end
