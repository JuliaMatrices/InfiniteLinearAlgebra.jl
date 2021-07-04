using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, InfiniteArrays, MatrixFactorizations, LazyArrays,
        FillArrays, SpecialFunctions, Test, SemiseparableMatrices, LazyBandedMatrices, BlockArrays
import LazyArrays: colsupport, rowsupport, MemoryLayout, DenseColumnMajor, TriangularLayout, resizedata!, arguments
import LazyBandedMatrices: BroadcastBandedLayout, InvDiagTrav, BroadcastBandedBlockBandedLayout
import BandedMatrices: _BandedMatrix, _banded_qr!, BandedColumns
import InfiniteLinearAlgebra: partialqr!, AdaptiveQRData, AdaptiveLayout, adaptiveqr
import SemiseparableMatrices: AlmostBandedLayout, VcatAlmostBandedLayout


@testset "Adaptive QR" begin
    @testset "banded" begin
        @testset "test partialqr!" begin
            A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ℵ₀, 1, 1)
            l,u = bandwidths(A)
            F = AdaptiveQRData(A);
            @test bandwidths(F.data) == (1,2)
            n = 3
            partialqr!(F,n);
            F̃ = qrunblocked(Matrix(A[1:n+l,1:n]))
            @test F̃.factors ≈ F.data[1:n+l,1:n]
            @test F̃.τ  ≈ F.τ
            @test triu!(F.data[1:n+1,1:n+2]) ≈ F̃.Q'*A[1:n+1,1:n+2] # test extra columns have been modified
            n  = 6
            partialqr!(F,n);
            F̃ = qrunblocked(Matrix(A[1:n+l,1:n]))
            @test F̃.factors ≈ F.data[1:n+l,1:n]
            @test F̃.τ  ≈ F.τ
            @test triu!(F.data[1:n+1,1:n+2]) ≈ F̃.Q'*A[1:n+1,1:n+2] # test extra columns have been modified
        end

        @testset "AdaptiveQRFactors" begin
            A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ℵ₀, 1, 1)
            F = qr(A);
            @test F.factors[1,1] ≈ -sqrt(2)
            @test F.factors[100,100] ≈ qrunblocked(A[1:101,1:100]).factors[100,100]
            @test F.τ[1] ≈ 1+sqrt(2)/2
            @test F.τ[100] ≈ qrunblocked(A[1:101,1:100]).τ[100]
            Q,R = F;
            @test bandwidths(R) == (0,2)
            @test size(R) == (∞,∞)
            @test R[1,1] == -sqrt(2)
        end

        @testset "col/rowsupport" begin
            A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ℵ₀, 1, 1)
            F = qr(A);
            @test MemoryLayout(typeof(F.factors)) isa AdaptiveLayout{BandedColumns{DenseColumnMajor}}
            @test bandwidths(F.factors) == (1,2)
            @test colsupport(F.factors,1) ==  1:2
            @test colsupport(F.factors,5) ==  3:6
            @test rowsupport(F.factors,1) ==  1:3
            @test rowsupport(F.factors,5) ==  4:7
            Q,R = F;
            @test MemoryLayout(typeof(R)) isa TriangularLayout
            @test colsupport(R,1) == 1:1
            @test colsupport(R,5) == 3:5
            @test rowsupport(R,1) == 1:3
            @test rowsupport(R,5) == 5:7
            @test colsupport(Q,1) == 1:2
            @test colsupport(Q,5) == 3:6
            @test rowsupport(Q,1) == 1:3
            @test rowsupport(Q,5) == 4:7
        end

        @testset "Qmul" begin
            A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ℵ₀, 1, 1)
            Q,R = qr(A);
            b = Vcat([1.,2,3],Zeros(∞))
            @test colsupport(Q,1:3) == colsupport(Q.factors,1:3) == 1:4
            @test lmul!(Q, Base.copymutable(b)).datasize[1] == 4
            @test lmul!(Q, Base.copymutable(b)).data[1:4] ≈ qr(A[1:4,1:3]).Q*[1,2,3]

            @test Q[1,1] ≈ -1/sqrt(2)
            @test Q[200_000,200_000] ≈ -1.0
            @test Q[1:101,1:100] ≈ qr(A[1:101,1:100]).Q[:,1:100]

            @test lmul!(Q, Base.copymutable(b)) == Q*b

            r = lmul!(Q', Base.copymutable(b))
            nr = length(r.data)
            @test qr(A[1:nr+1,1:nr]).Q'b[1:nr+1] ≈ r[1:nr+1]

            @test Q'*b == r

            @test factorize(A) isa typeof(qr(A))
            @test qr(A)\b == A\b
            @test (A*(A\b))[1:100] ≈ [1:3; Zeros(97)]

            @test Q * view(b,:) == Q*b
            @test Q' * view(b,:) == Q'*b
        end

        @testset "Bessel J" begin
            z = 1000; # the bigger z the longer before we see convergence
            A = BandedMatrix(0 => -2*(0:∞)/z, 1 => Ones(∞), -1 => Ones(∞))
            b = Vcat([besselj(1,z)], Zeros(∞))
            F = qr(A)
            @test qr(A[1:3000,1:3000]).Q'b[1:3000] ≈ (F.Q'b)[1:3000]
            @time J = A \ Vcat([besselj(1,z)], Zeros(∞))
            @test J[1:2000] ≈ [besselj(k,z) for k=0:1999]

            z = 10_000; # the bigger z the longer before we see convergence
            A = BandedMatrix(0 => -2*(0:∞)/z, 1 => Ones(∞), -1 => Ones(∞))
            @time J = A \ Vcat([besselj(1,z)], Zeros(∞))
            @test J[1:20_000] ≈ [besselj(k,z) for k=0:20_000-1]

            # Tridiagonal works too
            A =  LazyBandedMatrices.Tridiagonal(Ones(∞), -2*(0:∞)/z, Ones(∞))
            @test factorize(A) isa MatrixFactorizations.QR
            @time J = A \ [besselj(1,z); Zeros(∞)]
            @test J[1:20_000] ≈ [besselj(k,z) for k=0:20_000-1]
        end

        @testset "5-band" begin
            A = BandedMatrix(-2 => Ones(∞), -1 => Vcat(1, Zeros(∞)), 0 => Vcat([1,2,3],Zeros(∞)).+3, 1 => Vcat(1, Zeros(∞)), 2 => Ones(∞))
            b = Vcat([3,4,5],Zeros(∞))
            x = qr(A) \ b
            @test x[1:2000] ≈ (A[1:2000,1:2000]\b[1:2000])
        end

        @testset "broadcast" begin
            A = BandedMatrix(0 => -2*(0:∞), 1 => Ones(∞), -1 => Ones(∞))
            B = BandedMatrix(-2 => Ones(∞), -1 => Vcat(1, Zeros(∞)), 0 => Vcat([1,2,3],Zeros(∞)).+3, 1 => Vcat(1, Zeros(∞)), 2 => Ones(∞))

            AB = BroadcastArray(+,A,B)
            @test MemoryLayout(typeof(AB)) isa BroadcastBandedLayout{typeof(+)}

            C = cache(AB);
            resizedata!(C,103,100); resizedata!(C,203,200);
            @test C[103,104] ≈ 1.0
            C = qr(AB).factors.data.data;
            resizedata!(C,102,104); resizedata!(C,202,204);
            @test C[202,204] == AB[202,204]
            F = qr(AB);
            partialqr!(F.factors.data, 100); partialqr!(F.factors.data, 200);
            @test norm(F.factors.data.data.data[Base.OneTo.(F.factors.data.data.datasize)...]) ≤ 4000
            b = Vcat([3,4,5],Zeros(∞))
            @time x = qr(AB) \ b;
            @test x[1:300] ≈ AB[1:300,1:300] \ b[1:300]
        end

        @testset "triangular infqr" begin
            A = BandedMatrix(0 => 1:∞, 2 => Ones(∞))
            F = qr(A)
            @test F.Q[1:10,1:10] == Eye(10)
            @test F.R[1:10,1:10] == A[1:10,1:10]
        end

        @testset "diag special case" begin
            A = _BandedMatrix((1:∞)', ℵ₀, 0, 0)
            b = [[1,2,3]; zeros(∞)]
            @test A \ b == [ones(3); zeros(∞)]
        end

        @testset "Symmetric" begin
            A = Symmetric(BandedMatrix(0 => 1:∞, 1=> Ones(∞)))
            Ã = BandedMatrix(0 => 1:∞, 1=> Ones(∞), -1=> Ones(∞))
            @test qr(A).R[1:10,1:10] ≈ qr(Ã).R[1:10,1:10]
        end
    end

    @testset "almost-banded" begin
        @testset "one-band" begin
            A = Vcat(Ones(1,∞), BandedMatrix(0 => -Ones(∞), 1 => 1:∞))
            @test MemoryLayout(typeof(A)) isa VcatAlmostBandedLayout
            V = view(A,1:10,1:10)
            @test MemoryLayout(typeof(V)) isa VcatAlmostBandedLayout
            @test A[1:10,1:10] isa AlmostBandedMatrix
            @test AlmostBandedMatrix(V) == Matrix(V) == A[1:10,1:10]

            C = cache(A);
            @test C[1000,1000] ≡ 999.0
            F = adaptiveqr(A);
            partialqr!(F.factors.data,2);
            @test F.factors.data.data[1:3,1:5] ≈ qr(A[1:3,1:5]).factors
            partialqr!(F.factors.data,3);
            @test F.factors.data.data[1:4,1:6] ≈ qr(A[1:4,1:6]).factors

            F = adaptiveqr(A);
            partialqr!(F.factors.data,10);
            @test F.factors[1:11,1:10] ≈ qr(A[1:11,1:10]).factors
            @test F.τ[1:10] ≈ qr(A[1:11,1:10]).τ
            partialqr!(F.factors.data,20);
            @test F.factors[1:21,1:20] ≈ qr(A[1:21,1:20]).factors

            @test adaptiveqr(A).R[1:10,1:10] ≈ qr(A[1:11,1:10]).R

            @test qr(A) isa MatrixFactorizations.QR{Float64,<:InfiniteLinearAlgebra.AdaptiveQRFactors}
            @test factorize(A) isa MatrixFactorizations.QR{Float64,<:InfiniteLinearAlgebra.AdaptiveQRFactors}

            @test (adaptiveqr(A) \ [ℯ; zeros(∞)])[1:1000] ≈ (qr(A) \ [ℯ; zeros(∞)])[1:1000] ≈ (A \ [ℯ; zeros(∞)])[1:1000] ≈ [1/factorial(1.0k) for k=0:999]
        end

        @testset "two-bands" begin
            B = BandedMatrix(0 => -Ones(∞), 2 => (1:∞).* (2:∞))
            A = Vcat(Vcat(Ones(1,∞), ((-1).^(0:∞))'), B)
            @test MemoryLayout(typeof(A)) isa VcatAlmostBandedLayout

            @test qr(A) isa MatrixFactorizations.QR{Float64,<:InfiniteLinearAlgebra.AdaptiveQRFactors}
            @test factorize(A) isa MatrixFactorizations.QR{Float64,<:InfiniteLinearAlgebra.AdaptiveQRFactors}
            u = qr(A) \ [1; zeros(∞)]
            x = 0.1
            @test (exp(1 - x)*(-1 + exp(2 + 2x)))/(-1 + exp(4)) ≈ dot(u[1:1000], x.^(0:999))
            u = qr(A) \ Vcat([ℯ,1/ℯ], zeros(∞))
            @test u[1:1000] ≈ [1/factorial(1.0k) for k=0:999]
            u = qr(A) \ Vcat(ℯ,1/ℯ, zeros(∞))
            @test u[1:1000] ≈ [1/factorial(1.0k) for k=0:999]
            u = qr(A) \ [ℯ; 1/ℯ; zeros(∞)]
            @test u[1:1000] ≈ [1/factorial(1.0k) for k=0:999]

            A = Vcat(Ones(1,∞), ((-1.0).^(0:∞))', B)
            @test MemoryLayout(typeof(A)) isa VcatAlmostBandedLayout
            u = A \ Vcat(ℯ,1/ℯ, zeros(∞))
            @test u[1:1000] ≈ [1/factorial(1.0k) for k=0:999]
            u = qr(A) \ [ℯ; 1/ℯ; zeros(∞)]
            @test u[1:1000] ≈ [1/factorial(1.0k) for k=0:999]

            A = Vcat(Ones(1,∞), ((-1).^(0:∞))', B)
            u = A \ [ℯ; 1/ℯ; zeros(∞)]
            @test u[1:1000] ≈ [1/factorial(1.0k) for k=0:999]
        end

        @testset "more bands" begin
            L = Vcat(Ones(1,∞), ((-1).^(0:∞))',
                     BandedMatrix(-1 => Ones(∞), 1 => Ones(∞), 2 => 4:2:∞, 3 => Ones(∞), 5 => Ones(∞)))
            F = qr(L).factors.data;
            resizedata!(F.data,13,19)
            @test F.data.data[2,8] == -1
            F = qr(L);
            partialqr!(F,10);
            @test F.factors[1:10,1:10] ≈ qr(L[1:13,1:10]).factors[1:10,1:10]
            @test qr(L).factors[1:10,1:10] ≈ qr(L[1:13,1:10]).factors[1:10,1:10]

            u = L \ [1; 2; zeros(∞)]
            @test L[1:1000,1:1000]*u[1:1000] ≈ [1; 2; zeros(998)]
        end
    end

    @testset "block-banded" begin
        Δ = BandedMatrix(1 => Ones(∞), -1 => Ones(∞))/2
        A = KronTrav(Δ - 2I, Eye(∞))
        F = qr(A);
        @test abs.(F.factors[1:15,1:10]) ≈ abs.(qr(A[1:15,1:10]).factors)

        @test (F.Q' * [1; zeros(∞)])[1:6] ≈ [-0.9701425001453321,0,-0.23386170701251197,0,0,-0.06193705069863463]
        @test (F.Q*[1;zeros(∞)])[1:6] ≈ [-0.9701425001453321,0,0.24253562503633297,0,0,0]

        u = F \ [1; zeros(∞)]
        @test blockisequal(axes(A,2),axes(u,1))
        @test (A*u)[1:10] ≈ [1; zeros(9)]

        x = 0.1
        θ = acos(x)
        @test dot(u[getindex.(Block.(1:50),1:50)], sin.((1:50) .* θ)/sin(θ)) ≈ 1/(x-2)

        B = KronTrav(Eye(∞), Δ - 2I)
        u = B \ [1; zeros(∞)]

        @test dot(u[getindex.(Block.(1:50),1)], sin.((1:50) .* θ)/sin(θ)) ≈ 1/(x-2)

        L = A+B;
        @test MemoryLayout(L) isa BroadcastBandedBlockBandedLayout{typeof(+)}
        V = view(L,Block.(1:400),Block.(1:400))
        @test blockbandwidths(V) == blockbandwidths(L)
        @test subblockbandwidths(V) == blockbandwidths(L)
        @time u = L \ [1;zeros(∞)]

        x,y = 0.1,0.2
        θ,φ = acos(x),acos(y)
        @test u[Block.(1:50)] isa PseudoBlockArray
        @test (sin.((1:50) .* φ)/sin(φ))' * InvDiagTrav(u[Block.(1:50)]) * sin.((1:50) .* θ)/sin(θ) ≈ 1/(x+y-4)
        @test (L*u)[1:10] ≈ [1; zeros(9)]

        X = KronTrav(Δ,Eye(∞))
        Y = KronTrav(Eye(∞),Δ)
        II = KronTrav(Eye(∞),Eye(∞))
        @test MemoryLayout(2X + Y - 4II) isa BroadcastBandedBlockBandedLayout
        u =  (2X + Y - 8II) \ [1; zeros(∞)]
        x,y = 0.1,0.2
        θ,φ = acos(x),acos(y)
        @test (sin.((1:100) .* φ)/sin(φ))' * InvDiagTrav(u[Block.(1:100)]) * sin.((1:100) .* θ)/sin(θ) ≈ 1/(2x+y-8)
    end

    @testset "SymTridiagonal Toeplitz" begin
        Δ = SymTridiagonal(Fill(-2,∞), Fill(1,∞))
        h = 0.01
        A = I - h*Δ
        b = [1; 2; 3; zeros(∞)]
        @test (qr(A) \ b) ≈ (ul(A) \ b)
    end

    @testset "rdiv!" begin
        L = BandedMatrix(-1 => Ones(∞), 0 => Ones(∞))

        [1 Zeros(1,∞)] / LowerTriangular(L)
    end
end
