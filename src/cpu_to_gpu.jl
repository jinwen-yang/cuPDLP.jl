mutable struct CuLinearProgrammingProblem
    num_variables::Int64
    num_constraints::Int64
    variable_lower_bound::CuVector{Float64}
    variable_upper_bound::CuVector{Float64}
    isfinite_variable_lower_bound::CuVector{Bool}
    isfinite_variable_upper_bound::CuVector{Bool}
    objective_vector::CuVector{Float64}
    objective_constant::Float64
    constraint_matrix::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}
    constraint_matrix_t::CUDA.CUSPARSE.CuSparseMatrixCSR{Float64,Int32}
    right_hand_side::CuVector{Float64}
    num_equalities::Int64
end

mutable struct CuScaledQpProblem
    original_qp::CuLinearProgrammingProblem
    scaled_qp::CuLinearProgrammingProblem
    constraint_rescaling::CuVector{Float64}
    variable_rescaling::CuVector{Float64}
end

"""
Transfer quadratic program from CPU to GPU
"""
function qp_cpu_to_gpu(problem::QuadraticProgrammingProblem)
    num_constraints, num_variables = size(problem.constraint_matrix)
    isfinite_variable_lower_bound = Vector{Bool}(isfinite.(problem.variable_lower_bound))
    isfinite_variable_upper_bound = Vector{Bool}(isfinite.(problem.variable_upper_bound))

    d_variable_lower_bound = CuArray{Float64}(undef, num_variables)
    d_variable_upper_bound = CuArray{Float64}(undef, num_variables)
    d_isfinite_variable_lower_bound = CuArray{Bool}(undef, num_variables)
    d_isfinite_variable_upper_bound = CuArray{Bool}(undef, num_variables)
    d_objective_vector = CuArray{Float64}(undef, num_variables)
    d_right_hand_side = CuArray{Float64}(undef, num_constraints)

    copyto!(d_variable_lower_bound, problem.variable_lower_bound)
    copyto!(d_variable_upper_bound, problem.variable_upper_bound)
    copyto!(d_isfinite_variable_lower_bound, isfinite_variable_lower_bound)
    copyto!(d_isfinite_variable_upper_bound, isfinite_variable_upper_bound)
    copyto!(d_objective_vector, problem.objective_vector)
    copyto!(d_right_hand_side, problem.right_hand_side)

    d_constraint_matrix = CUDA.CUSPARSE.CuSparseMatrixCSR(problem.constraint_matrix)
    d_constraint_matrix_t = CUDA.CUSPARSE.CuSparseMatrixCSR(problem.constraint_matrix')

    return CuLinearProgrammingProblem(
        num_variables,
        num_constraints,
        d_variable_lower_bound,
        d_variable_upper_bound,
        d_isfinite_variable_lower_bound,
        d_isfinite_variable_upper_bound,
        d_objective_vector,
        problem.objective_constant,
        d_constraint_matrix,
        d_constraint_matrix_t,
        d_right_hand_side,
        problem.num_equalities,
    )
end


"""
Transfer scaled QP from CPU to GPU
"""
function scaledqp_cpu_to_gpu(scaled_problem::ScaledQpProblem)
    d_constraint_rescaling = CuArray{Float64}(undef,length(scaled_problem.constraint_rescaling))
    d_variable_rescaling = CuArray{Float64}(undef,length(scaled_problem.variable_rescaling))

    copyto!(d_constraint_rescaling, scaled_problem.constraint_rescaling)
    copyto!(d_variable_rescaling, scaled_problem.variable_rescaling)

    return CuScaledQpProblem(
        qp_cpu_to_gpu(scaled_problem.original_qp),
        qp_cpu_to_gpu(scaled_problem.scaled_qp),
        d_constraint_rescaling,
        d_variable_rescaling,
    )
end

"""
Transfer solutions from GPU to CPU
"""
function gpu_to_cpu!(
    d_primal_solution::CuVector{Float64},
    d_dual_solution::CuVector{Float64},
    primal_solution::Vector{Float64},
    dual_solution::Vector{Float64},
)
    copyto!(primal_solution, d_primal_solution)
    copyto!(dual_solution, d_dual_solution)
end


