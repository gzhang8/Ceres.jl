module Ceres

using CeresCApi_jll
import Libdl


include("libceres.jl")
include("loss_function.jl")

using ForwardDiff

export Problem, AddResidualBlock!, solve, CreateQuaternionParameterization,
        SetLocalParameterization!, SetParameterBlockConstant!, # CostFunction,

        AngleAxisRotatePoint, two_pose_point3d_dist_error, SE3_edge_error,
        reprojection_error,

        R2quat, quat2R, quaternion_parameterization,

        # loss
        LossFunction, EmptyLoss, HuberLoss

quaternion_parameterization = nothing

function __init__()

    h = CeresCApi_jll.libceres_c_api_handle
    ccall(Libdl.dlsym(h, :ceres_init), Cvoid, ())

    quat_param_ptr = ccall(
        Libdl.dlsym(h, :ceres_create_quaternion_parameterization),
        Ptr{Cvoid},
        ()
    )

    global quaternion_parameterization = QuaternionParameterization(quat_param_ptr)

end

```
This is for that ForwardDiff can only take vector of primitive types
```
function recover_params_shape(params_vec::Vector, params_sizes::Vector{Int64})
    parameters::Vector{SubArray} = Vector{SubArray}(undef, 0)
    sidx = 1
    for i in 1:length(params_sizes)
        push!(parameters, view(params_vec, sidx:sidx + params_sizes[i] - 1))
        sidx = sidx + params_sizes[i]
    end
    return parameters
end

function cost_function_autodiff(func,
                                user_data::Vector{Float64},
                                parameters::Vector{Vector{Float64}},
                                residuals::Vector{Float64},
                                jacobians::Vector{Vector{Float64}})::Int32

    resres = func(user_data, parameters)
    for idx = 1:length(residuals)
        residuals[idx] = resres[idx]
    end

    params_sizes = [length(v) for v in parameters]
    params_vec = reduce(vcat, parameters)
    f_params = (params) -> func(user_data, recover_params_shape(params, params_sizes))

    J = ForwardDiff.jacobian(f_params, params_vec)
    # , ForwardDiff.JacobianConfig(f_params, parameters)

    if (length(jacobians) == 0)
        return true;
    end

    sidx = 1
    for i = 1:length(jacobians)
      # we need a row major jacobian
        jvec = vec(view(J, :, sidx:sidx + params_sizes[i] - 1)')
        if !((length(jacobians[i]) == 0) | (length(jacobians[i]) == length(jvec)))
            println("jacobian vector size wrong")
        end
        for j = 1:length(jacobians[i])
            jacobians[i][j] = jvec[j]
        end
        sidx = sidx + params_sizes[i]
    end
    return true;
end

# // (double* user_data,
# //  double** parameters,
# //  double* residuals,
# //  double** jacobians,
# //  int user_data_size,
# //  int num_residuals,
# //  int num_parameter_block,
# //  int* parameter_block_sizes)
"""
typedef int (*ceres_cost_function_t)(void* user_data,
                                     double** parameters,
                                     double* residuals,
                                     double** jacobians);
"""
function costfunctionwrap(cost_function_jl,
                        user_data_ptr::Ptr{Cdouble},
                        params_pptr::Ptr{Ptr{Cdouble}},
                        residuals_ptr::Ptr{Cdouble},
                        jacobians_pptr::Ptr{Ptr{Cdouble}},
                        user_data_size::Cint,
                        num_residuals::Cint,
                        num_parameter_block::Cint,
                        parameter_block_sizes_ptr::Ptr{Cint})::Cint

    user_data = unsafe_wrap(Array, user_data_ptr, user_data_size, own=false)

    pbsizes = unsafe_wrap(Array, parameter_block_sizes_ptr, num_parameter_block, own=false)
    params_ptr = unsafe_wrap(Array, params_pptr, num_parameter_block, own=false)
    params = Vector{Vector{Float64}}(undef, 0)

    jacobians_ptr = Vector{Ptr{Cdouble}}(undef, num_parameter_block)
    fill!(jacobians_ptr, C_NULL)
    if (jacobians_pptr != C_NULL)
        jacobians_ptr = unsafe_wrap(Array, jacobians_pptr, num_parameter_block, own=false)
    end

    jacobians = Vector{Vector{Float64}}(undef, num_parameter_block)
    fill!(jacobians, Vector{Float64}(undef, 0))

    for i = 1:num_parameter_block
        push!(params, unsafe_wrap(Array, params_ptr[i], pbsizes[i], own=false))
        if jacobians_ptr[i] != C_NULL
            jacobians[i] = unsafe_wrap(Array, jacobians_ptr[i],
                                     pbsizes[i] * num_residuals, own=false)
        end
    end

    residuals = unsafe_wrap(Array, residuals_ptr, num_residuals, own=false)

    return cost_function_autodiff(cost_function_jl, user_data, params, residuals, jacobians)
end


struct Problem
    c_ptr::Ptr{Cvoid}
    # costfunctions::Vector{CostFunction}

    # prevent GC collect back params ptr vector
    params_ptr_vec::Vector{Vector{Ptr{Cdouble}}}
    cost_function_data::Vector{Vector{Float64}}
    params_sizes_int32::Vector{Vector{Cint}}
end

function Problem()
    problemc = createproblem_c()

    Problem(problemc,
      Vector{Vector{Ptr{Cdouble}}}(undef, 0),
      Vector{Vector{Float64}}(undef, 0),
      Vector{Vector{Cint}}(undef, 0))
end



function AddResidualBlock!(problem::Problem,
                           cost_function_jl,
                           cost_function_data::Vector{Float64},
                           params::Vector{Vector{Float64}},
                           residuals_num::Int64;
                           loss_function::TL=EmptyLoss()
) where TL <: LossFunction
    params_sizes_int32 = [Cint(length(p)) for p in params]
    pbsizes_ptr = Base.unsafe_convert(Ptr{Cint}, params_sizes_int32)
    # pb_sizes_ptr = Base.unsafe_convert(Ptr{Cint}, params_sizes)
    push!(problem.params_sizes_int32, params_sizes_int32)

    params_ptr = Vector{Ptr{Cdouble}}(undef, 0)
    for i = 1:length(params)
       # User must protect params from GC
        push!(params_ptr, Base.unsafe_convert(Ptr{Cdouble}, params[i]))
    end

    params_pptr = Base.unsafe_convert(Ptr{Ptr{Cdouble}}, params_ptr)
    push!(problem.params_ptr_vec, params_ptr)


    dataptr = Base.unsafe_convert(Ptr{Cdouble}, cost_function_data)
    push!(problem.cost_function_data, cost_function_data)
    # push!(problem.costfunctions, costfunction)
    ############################################################################
    # params_sizes_int32 = convert(Vector{Int32}, params_sizes)

    lambdaf = (user_data_ptr::Ptr{Cdouble},
               params_pptr::Ptr{Ptr{Cdouble}},
               residuals_ptr::Ptr{Cdouble},
               jacobians_pptr::Ptr{Ptr{Cdouble}},
               user_data_size::Cint,
               num_residuals::Cint,
               num_parameter_block::Cint,
               parameter_block_sizes_ptr::Ptr{Cint}) -> costfunctionwrap(
                              cost_function_jl,
                              user_data_ptr,
                              params_pptr,
                              residuals_ptr,
                              jacobians_pptr,
                              user_data_size,
                              num_residuals,
                              num_parameter_block,
                              parameter_block_sizes_ptr)
    costcb_c = @cfunction($lambdaf,
            Cint,
            (Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Cint, Cint, Ptr{Cint}))

    loss_function_ptr = if TL == EmptyLoss
        C_NULL
    else
        @assert loss_function.c_ptr != C_NULL
        @show loss_function.c_ptr
        loss_function.c_ptr
    end

    val = addresidualblock_c(problem.c_ptr,
                            costcb_c.ptr, # cfunction
                            dataptr,
                            convert(Cint, length(cost_function_data)),
                            loss_function_ptr,  # loss function c ptr
                            convert(Ptr{Cdouble}, C_NULL),
                            convert(Cint, residuals_num),
                            convert(Cint, length(params)),
                            pbsizes_ptr,
                            params_pptr)
    return val

    ############################################################################

end


################# local paramterization ####################


abstract type LocalParameterization end

struct QuaternionParameterization <: LocalParameterization
    c_ptr::Ptr{Cvoid}
end


function SetLocalParameterization!(problem::Problem,
                                   params::Vector{Float64},
                                   local_parameterization::QuaternionParameterization)# where T<:

    param_ptr = Base.unsafe_convert(Ptr{Cdouble}, params)
    set_local_parameterization_c(problem.c_ptr, param_ptr, local_parameterization.c_ptr)
end

function CreateQuaternionParameterization()
    quater_c_ptr = create_quaternion_parameterization_c()
    return QuaternionParameterization(quater_c_ptr)
end


######################### solve ###############################

function solve(problem::Problem; max_iter_num::Int64=100, solver_type::Int64=1)
    solve_c(problem.c_ptr, max_iter_num, solver_type)
end

include("parameter_block.jl")



end # module
