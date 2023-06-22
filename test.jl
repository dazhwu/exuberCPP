using Random
using Dates
using DataFrames
using Plots
using CSV
#using HypothesisTests
using BenchmarkTools



include("exuberCPP.jl")
#import ._exuber

df_long = CSV.read("/home/rstudio/gsadf_CPP/ori.csv", DataFrame)
#df_long = CSV.read("./data/bubbles.csv", DataFrame)

#df_long = CSV.read("./data/growth.csv", DataFrame)
df_wide = unstack(df_long, :date, :msa, :ratio)
df_wide = df_wide[completecases(df_wide), :]
#CSV.write("test_exuber.csv", df_wide)

bootstrap = :psy  #:both  #:wm  #:psy
nboot = 1000
start_time = now()
adflag=1
df_mat::Matrix{Float64}= Matrix(df_wide[:, 2:end])
obs, num_cols = size(df_mat)
start_time = now()
bsadf, cv, wm, Min_Window = _exuber.exuber(df_mat, adflag, 0, bootstrap, nboot)
d_label = df_wide[:, 1][Min_Window+adflag+1:obs]
println("Time elapsed: ", now() - start_time)



@time a = _exuber.CV_PSY(1000, 100, 0, 1)

