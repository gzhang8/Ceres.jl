

# """
# void ceres_init(){
# """
# function ceres_init()
#     @show "init ing ceres!"
#     ccall((:ceres_init, :libceres_c_api), Cvoid, ())
# end

function createproblem_c()::Ptr{Cvoid}
    return ccall((:ceres_create_problem, "libceres_c_api"), Ptr{Cvoid}, ())
end


function solve_c(problem, max_iter_num::Int64, solver_type::Int64)
    ccall( (:ceres_solve, "libceres_c_api"),
        Cvoid,
        (Ptr{Cvoid}, Cint, Cint),
        problem, convert(Cint, max_iter_num),
        convert(Cint, solver_type))
end

function freeproblem_c(problem)
    ccall((:ceres_free_problem, "libceres_c_api"), Cvoid, (Ptr{Cvoid},), problem)
end



"""
# void* ceres_problem_add_residual_block(
#    void* problem,
#    void* cost_function,
#    double* cost_function_data,
#    int cost_function_data_size,
#    void* loss_function,
#    double* loss_function_data,
#    int num_residuals,
#    int num_parameter_blocks,
#    int* parameter_block_sizes,
#    double** parameters);

lossfunction::Ptr{Cvoid},  # now is ceres c loss. It could be made to use
 cfunction in the future
"""
function addresidualblock_c(problem::Ptr{Cvoid},
                            costfunction::Ptr{Cvoid}, # cfunction
                            data::Ptr{Cdouble},
                            cost_function_data_size::Cint,
                            lossfunction::Ptr{Cvoid},  # ceres c loss
                            lossfunctionuserdata::Ptr{Cdouble},
                            numresiduals::Cint,
                            numparamblocks::Cint,
                            parametersizes::Ptr{Cint},
                            parameterpointers::Ptr{Ptr{Cdouble}})

    # void* ceres_problem_add_residual_block(
    #    void* problem,
    #    void* cost_function,
    #    double* cost_function_data,
    #    int cost_function_data_size,
    #    void* loss_function,
    #    double* loss_function_data,
    #    int num_residuals,
    #    int num_parameter_blocks,
    #    int* parameter_block_sizes,
    #    double** parameters);
    val = ccall( (:ceres_problem_add_residual_block, "libceres_c_api"),
              Ptr{Cvoid},
              (Ptr{Cvoid},            # ceres_problem_t* problem,
               Ptr{Cvoid},            # ceres_cost_function_t cost_function,
               Ptr{Cdouble},          # void* cost_function_data,
               Cint,                  # int cost_function_data_size,
               Ptr{Cvoid},            # ceres_loss_function_t loss_function,
               Ptr{Cdouble},          # void* loss_function_data,
               Cint,                  # int num_residuals,
               Cint,                  # int num_parameter_blocks,
               Ptr{Cint},             # int* parameter_block_sizes,
               Ptr{Ptr{Cdouble}}),    # double** parameters
              problem,
              costfunction,
              data,
              cost_function_data_size,
              lossfunction,
              lossfunctionuserdata,
              numresiduals,
              numparamblocks,
              parametersizes,
              parameterpointers )
    return val
end


function set_local_parameterization_c(problem::Ptr{Cvoid}, # why uint8? see line 6
                                      parameters::Ptr{Cdouble},
                                      local_parameterization::Ptr{Cvoid})
    ccall( (:ceres_problem_set_parameterization, "libceres_c_api"),
                Cvoid,
                (Ptr{UInt8},            # ceres_problem_t* problem,
                 Ptr{Cdouble},            # ceres_cost_function_t cost_function,
                 Ptr{Cvoid}),          # void* cost_function_data,
                problem,
                parameters,
                local_parameterization)

end

"""
void* ceres_create_quaternion_parameterization();
"""
function create_quaternion_parameterization_c()
    ret = ccall( (:ceres_create_quaternion_parameterization, "libceres_c_api"),
               Ptr{Cvoid}, () )
    return ret
end

# void ceres_set_parameter_block_constant(void* problem, double* data){
function ceres_set_parameter_block_constant_c(problem::Ptr{Cvoid}, # why uint8? see line 6
                                              parameters::Ptr{Cdouble})
    ccall((:ceres_set_parameter_block_constant, "libceres_c_api"),
          Cvoid, (Ptr{Cvoid},            # ceres_problem_t* problem,
                 Ptr{Cdouble}),
           problem,
           parameters)
end

# void ceres_set_parameter_block_variable(void* c_problem, double *values) {
function ceres_set_parameter_block_variable_c(problem::Ptr{Cvoid}, # why uint8? see line 6
                                              parameters::Ptr{Cdouble})
    ccall((:ceres_set_parameter_block_variable, "libceres_c_api"),
          Cvoid, (Ptr{Cvoid},            # ceres_problem_t* problem,
                 Ptr{Cdouble}),
           problem,
           parameters)
end

# void* ceres_create_huber_loss(double delta){
function ceres_create_huber_loss_c(δ::Float64)::Ptr{Cvoid}
    ccall((:ceres_create_huber_loss, "libceres_c_api"),
           Ptr{Cvoid}, (Cdouble,), δ)
end


# // bounds
#
# void ceres_SetParameterLowerBound(void* c_problem, double *values, int index, double lower_bound) {
#     ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);
#     problem->SetParameterLowerBound(values, index, lower_bound);
# }

function ceres_set_parameter_lower_bound_c(
      problem::Ptr{Cvoid}, # why uint8? see line 6
      parameters::Ptr{Cdouble},
      index_jl::Union{Int32,Int64},
      lower_bound::Float64
)
    ccall((:ceres_SetParameterLowerBound, "libceres_c_api"),
          Cvoid, (Ptr{Cvoid},            # ceres_problem_t* problem,
                  Ptr{Cdouble},
                  Cint,
                  Cdouble),
           problem,
           parameters,
           Cint(index_jl - 1),
           Cdouble(lower_bound))
end


# void ceres_SetParameterUpperBound(void* c_problem, double *values, int index, double upper_bound) {
#      ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);
#      problem->SetParameterUpperBound(values, index, upper_bound);
# }
function ceres_set_parameter_upper_bound_c(
      problem::Ptr{Cvoid}, 
      parameters::Ptr{Cdouble},
      index_jl::Union{Int32,Int64},
      upper_bound::Float64
)
    ccall((:ceres_SetParameterUpperBound, "libceres_c_api"),
          Cvoid, (Ptr{Cvoid},            # ceres_problem* problem,
                  Ptr{Cdouble},
                  Cint,
                  Cdouble),
           problem,
           parameters,
           Cint(index_jl - 1),
           Cdouble(upper_bound))
end

# // constant
# void ceres_SetParameterBlockConstant(void* c_problem, double *values) {
#      ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);
#      problem->SetParameterBlockConstant(values);
# }
# void ceres_SetParameterBlockVariable(void* c_problem, double *values) {
#      ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);
#      problem->SetParameterBlockVariable(values);
# }
