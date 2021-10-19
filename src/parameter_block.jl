
function set_parameter_lower_bound!(
    p::Problem,
    block::Vector{Float64},
    index::Union{Int32, Int64},
    bound::Float64
)
    ceres_set_parameter_lower_bound_c(
        p.c_ptr,
        Base.unsafe_convert(Ptr{Cdouble}, block),
        index,
        bound
    )
end


function set_parameter_upper_bound!(
    p::Problem,
    block::Vector{Float64},
    index::Union{Int32, Int64},
    bound::Float64
)
    ceres_set_parameter_upper_bound_c(
        p.c_ptr,
        Base.unsafe_convert(Ptr{Cdouble}, block),
        index,
        bound
    )
end


function SetParameterBlockConstant!(problem::Problem,
                                    params::Vector{Float64})
    param_ptr = Base.unsafe_convert(Ptr{Cdouble}, params)
    ceres_set_parameter_block_constant_c(problem.c_ptr, param_ptr)
end


function SetParameterBlockVariable!(problem::Problem,
                                    params::Vector{Float64})
    param_ptr = Base.unsafe_convert(Ptr{Cdouble}, params)
    ceres_set_parameter_block_variable_c(problem.c_ptr, param_ptr)
end
