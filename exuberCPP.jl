__precompile__(true)
@everywhere module _exuber
using LinearAlgebra
using Statistics
using Random
using Distributed
using SharedArrays, LinearAlgebra
#println("Loading exuberCPP.jl")
export exuber, CV_PSY


const gsadf::String =ifelse(Sys.iswindows(),joinpath(@__DIR__, "gsadf.dll"), joinpath(@__DIR__, "gsadf.so"))



@inbounds function exuber(df::Matrix{Float64}, adflag::Int64, Min_Window::Int64=0, bootstrap=:psy, nboot::Int=500)

    obs, num_cols = size(df)

    #println(obs - Min_Window - adflag)
    Min_Window = ifelse(Min_Window == 0, default_win_size(obs), Min_Window)
    max_start::Int64 = obs - adflag - Min_Window

    #d_label = df[:, 1][Min_Window+adflag+1:obs]

    cv = Matrix{Float64}(undef, obs - Min_Window - adflag, 3)
    wm = Matrix{Float64}(undef, obs - Min_Window - adflag, num_cols)

    bsadf = Matrix{Float64}(undef, obs - Min_Window - adflag, num_cols)
    badf = Matrix{Float64}(undef, obs - Min_Window - adflag, num_cols)
    #local cv
    Threads.@threads for i in 1:num_cols
        ts = Vector{Float64}(@view(df[:, i]))
        #results = rls_gsadf(ts, adflag, Min_Window)
        temp_bsadf = Vector{Float64}(undef, obs - Min_Window - adflag)
        temp_badf = Vector{Float64}(undef, obs - Min_Window - adflag)
        size_ts::Int32 = length(ts)
        #@ccall "./gsadf.so".rls_gsadf(ts::Ptr{Cdouble}, size_ts::Cint, adflag::Cint, Min_Window::Cint, temp_badf::Ptr{Cdouble}, temp_bsadf::Ptr{Cdouble})::Cvoid
        ccall((:rls_gsadf, gsadf), Cvoid, (Ptr{Cdouble}, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}), ts, size_ts, adflag, Min_Window, temp_badf, temp_bsadf)

        #rls_gsadf(ts, adflag, Min_Window, temp_badf, temp_bsadf)
        
        bsadf[:, i] = temp_bsadf
        badf[:, i] = temp_badf

        # if bootstrap == :wm || bootstrap == :both
        #     wm[:, i-1] = @view wmboot(ts, Min_Window, adflag)[:, 2]
        # end
    end

    if bootstrap == :psy || bootstrap == :both
        #cv = CV_bsadf(obs, Min_Window, adflag, nboot)      #CV_PSY(T::Int, m::Int, swindow0::Int, adflag::Int)

        cv = CV_PSY(obs, nboot, Min_Window, adflag)
    end

    #return (bsadf, wm, d_label)

    return (bsadf, cv, wm, Min_Window)
end

function default_win_size(obs::Int64)
    r0 = 0.01 + 1.8 / sqrt(obs)
    return floor(Int64, r0 * obs)
end

@inbounds function CV_PSY(T::Int64, nboot::Int64, min_window::Int64, adflag::Int64)

    min_window = ifelse(min_window == 0, default_win_size(T), min_window)
    qe = [0.90, 0.95, 0.99]

    dim = T - adflag - min_window
    MPSY = SharedArray(zeros(Float64, dim, nboot))  # Matrix{Float64}(undef, dim, nboot)
    tbr = Matrix{Float64}(undef, dim, length(qe))
    badf = Vector{Float64}(undef, dim)
    bsadf = Vector{Float64}(undef, dim)
    seed = 123

    Random.seed!(seed)
    #e::Matrix{Float64} = randn(T, nboot) .+ 1.0 / T
    #e::Matrix{Float64} = randn(T-adflag, nboot) .+ 1.0 / T

    #y=cumsum(e, dims=1)
    #e = Vector{Float64}(undef, T - adflag)
    y = Vector{Float64}(undef, T - adflag)
    @sync @distributed for iter ∈ 1:nboot             
        y .= cumsum(randn(T - adflag) .+ 1.0 / T)                
        ccall((:rls_gsadf, gsadf), Cvoid, (Ptr{Cdouble}, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}), y, T - adflag, 0, min_window, badf, bsadf)

        MPSY[:, iter] = accumulate(max, badf)
        # if mod(iter, 100) == 0
        #     println(iter)
        # end
    end
    
    start::Int64 = nboot*qe[1]
    sort!(MPSY; dims=2, alg=PartialQuickSort(start:nboot))
    for j in eachindex(qe)
        position=Int64(nboot*qe[j])
        tbr[:, j] = MPSY[:, position]
    end

    # for i in 1:dim
    #     for j in eachindex(qe)
    #         tbr[i, j] = quantile(@view(MPSY[i, :]), qe[j])
    #     end
    # end

    return (tbr)
end

end
