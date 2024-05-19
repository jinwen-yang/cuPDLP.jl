
mutable struct CuDualStats
    dual_objective::Float64
    dual_residual::CuVector{Float64}
    reduced_costs::CuVector{Float64}
end

mutable struct BufferKKTState
    primal_solution::CuVector{Float64}
    dual_solution::CuVector{Float64}
    primal_product::CuVector{Float64}
    primal_gradient::CuVector{Float64}
    lower_variable_violation::CuVector{Float64}
    upper_variable_violation::CuVector{Float64}
    constraint_violation::CuVector{Float64}
    dual_objective_contribution_array::CuVector{Float64}
    reduced_costs_violation::CuVector{Float64}
    dual_stats::CuDualStats
    dual_res_inf::Float64
end


"""
Kernel to compute the violation of primal constraints
"""
function compute_primal_residual_constraint_kernel!(
    activities::CuDeviceVector{Float64},
    right_hand_side::CuDeviceVector{Float64},
    num_equalities::Int64,
    num_constraints::Int64,
    constraint_violation::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_equalities
        @inbounds begin
            constraint_violation[tx] = right_hand_side[tx] - activities[tx]
        end
    end
    if num_equalities + 1 <= tx <= num_constraints
        @inbounds begin
            constraint_violation[tx] = max(right_hand_side[tx] - activities[tx], 0.0)
        end
    end
    return 
end

"""
Kernel to compute the violation of primal variable bound
"""
function compute_primal_residual_variable_kernel!(
    primal_vec::CuDeviceVector{Float64},
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    num_variables::Int64,
    lower_variable_violation::CuDeviceVector{Float64},
    upper_variable_violation::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            lower_variable_violation[tx] = max(variable_lower_bound[tx] - primal_vec[tx], 0.0)
            upper_variable_violation[tx] = max(primal_vec[tx] - variable_upper_bound[tx], 0.0)
        end
    end
    return 
end

"""
Compute primal residual
"""
function compute_primal_residual!(
    problem::CuLinearProgrammingProblem,
    buffer_kkt::BufferKKTState,
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)
    NumBlockDual = ceil(Int64, problem.num_constraints/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal compute_primal_residual_variable_kernel!(
        buffer_kkt.primal_solution,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        problem.num_variables,
        buffer_kkt.lower_variable_violation,
        buffer_kkt.upper_variable_violation,
    )

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_primal_residual_constraint_kernel!(
        buffer_kkt.primal_product,
        problem.right_hand_side,
        problem.num_equalities,
        problem.num_constraints,
        buffer_kkt.constraint_violation,
    )
end
      
"""
Compute primal objective
"""
function primal_obj(
    problem::CuLinearProgrammingProblem,
    primal_solution::CuVector{Float64},
)
    return problem.objective_constant +
        CUDA.dot(problem.objective_vector, primal_solution)
end

"""
Kernel to compute the contribution of reduced costs to dual objective
"""
function reduced_costs_dual_objective_contribution_kernel!(
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    reduced_costs::CuDeviceVector{Float64},
    num_variables::Int64,
    dual_objective_contribution_array::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            if reduced_costs[tx] > 0.0
                dual_objective_contribution_array[tx] = variable_lower_bound[tx] * reduced_costs[tx]
            elseif reduced_costs[tx] < 0.0
                dual_objective_contribution_array[tx] = variable_upper_bound[tx] * reduced_costs[tx]
            else
                dual_objective_contribution_array[tx] = 0.0
            end
        end
    end
    return 
end

"""
Compute the contribution of reduced costs to dual objective
"""
function reduced_costs_dual_objective_contribution(
    problem::CuLinearProgrammingProblem,
    buffer_kkt::BufferKKTState,
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal reduced_costs_dual_objective_contribution_kernel!(
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        buffer_kkt.dual_stats.reduced_costs,
        problem.num_variables,
        buffer_kkt.dual_objective_contribution_array,
    )  
 
    dual_objective_contribution = sum(buffer_kkt.dual_objective_contribution_array)

    return dual_objective_contribution
end

"""
Kernel to compute the reduced costs from primal gradient
"""
function compute_reduced_costs_from_primal_gradient_kernel!(
    primal_gradient::CuDeviceVector{Float64},
    isfinite_variable_lower_bound::CuDeviceVector{Bool},
    isfinite_variable_upper_bound::CuDeviceVector{Bool},
    num_variables::Int64,
    reduced_costs::CuDeviceVector{Float64},
    reduced_costs_violation::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            reduced_costs[tx] = max(primal_gradient[tx], 0.0) * isfinite_variable_lower_bound[tx] + min(primal_gradient[tx], 0.0) * isfinite_variable_upper_bound[tx]

            reduced_costs_violation[tx] = primal_gradient[tx] - reduced_costs[tx]
        end
    end
    return 
end

"""
Compute reduced costs from primal gradient
"""
function compute_reduced_costs_from_primal_gradient!(
    problem::CuLinearProgrammingProblem,
    buffer_kkt::BufferKKTState,
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal compute_reduced_costs_from_primal_gradient_kernel!(
        buffer_kkt.primal_gradient,
        problem.isfinite_variable_lower_bound,
        problem.isfinite_variable_upper_bound,
        problem.num_variables,
        buffer_kkt.dual_stats.reduced_costs,
        buffer_kkt.reduced_costs_violation,
    )  
end

"""
Kernel to compute the dual residual
"""
function compute_dual_residual_kernel!(
    dual_solution::CuDeviceVector{Float64},
    num_equalities::Int64,
    num_inequalities::Int64,
    dual_residual::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_inequalities
        @inbounds begin
            dual_residual[tx] = max(-dual_solution[tx+num_equalities], 0.0)
        end
    end
    return 
end

"""
Compute the dual residual and dual objective
"""
function compute_dual_stats!(
    problem::CuLinearProgrammingProblem,
    buffer_kkt::BufferKKTState,
)
    compute_reduced_costs_from_primal_gradient!(problem, buffer_kkt)

    NumBlockIneq = ceil(Int64, (problem.num_constraints-problem.num_equalities)/ThreadPerBlock)

    if NumBlockIneq >= 1
        CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockIneq compute_dual_residual_kernel!(
            buffer_kkt.dual_solution,
            problem.num_equalities,
            problem.num_constraints - problem.num_equalities,
            buffer_kkt.dual_stats.dual_residual,
        )  
    end

    buffer_kkt.dual_res_inf = CUDA.norm([buffer_kkt.dual_stats.dual_residual; buffer_kkt.reduced_costs_violation], Inf)

    base_dual_objective = CUDA.dot(problem.right_hand_side, buffer_kkt.dual_solution) + problem.objective_constant

    buffer_kkt.dual_stats.dual_objective = base_dual_objective + reduced_costs_dual_objective_contribution(problem, buffer_kkt)
end

function corrected_dual_obj(buffer_kkt::BufferKKTState)
    if buffer_kkt.dual_res_inf == 0.0
        return buffer_kkt.dual_stats.dual_objective
    else
        return -Inf
    end
end

"""
Compute convergence information of the given primal and dual solutions
"""
function compute_convergence_information(
    problem::CuLinearProgrammingProblem,
    qp_cache::CachedQuadraticProgramInfo,
    primal_iterate::CuVector{Float64},
    dual_iterate::CuVector{Float64},
    eps_ratio::Float64,
    candidate_type::PointType,
    primal_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
    buffer_kkt::BufferKKTState,
)
    
    ## construct buffer_kkt
    buffer_kkt.primal_solution = primal_iterate
    buffer_kkt.dual_solution = dual_iterate
    buffer_kkt.primal_product = primal_product
    buffer_kkt.primal_gradient = primal_gradient
    

    convergence_info = ConvergenceInformation()

    compute_primal_residual!(problem, buffer_kkt)
    convergence_info.primal_objective = primal_obj(problem, buffer_kkt.primal_solution)
    convergence_info.l_inf_primal_residual = CUDA.norm([buffer_kkt.constraint_violation; buffer_kkt.lower_variable_violation; buffer_kkt.upper_variable_violation], Inf)
    convergence_info.l2_primal_residual = CUDA.norm([buffer_kkt.constraint_violation; buffer_kkt.lower_variable_violation; buffer_kkt.upper_variable_violation], 2)
    convergence_info.relative_l_inf_primal_residual =
        convergence_info.l_inf_primal_residual /
        (eps_ratio + qp_cache.l_inf_norm_primal_right_hand_side)
    convergence_info.relative_l2_primal_residual =
        convergence_info.l2_primal_residual /
        (eps_ratio + qp_cache.l2_norm_primal_right_hand_side)
    convergence_info.l_inf_primal_variable = CUDA.norm(buffer_kkt.primal_solution, Inf)
    convergence_info.l2_primal_variable = CUDA.norm(buffer_kkt.primal_solution, 2)

    compute_dual_stats!(problem, buffer_kkt)
    convergence_info.dual_objective = buffer_kkt.dual_stats.dual_objective
    convergence_info.l_inf_dual_residual = buffer_kkt.dual_res_inf
    convergence_info.l2_dual_residual = norm([buffer_kkt.dual_stats.dual_residual; buffer_kkt.reduced_costs_violation], 2)
    convergence_info.relative_l_inf_dual_residual =
        convergence_info.l_inf_dual_residual /
        (eps_ratio + qp_cache.l_inf_norm_primal_linear_objective)
    convergence_info.relative_l2_dual_residual =
        convergence_info.l2_dual_residual /
        (eps_ratio + qp_cache.l2_norm_primal_linear_objective)
    convergence_info.l_inf_dual_variable = CUDA.norm(buffer_kkt.dual_solution, Inf)
    convergence_info.l2_dual_variable = CUDA.norm(buffer_kkt.dual_solution, 2)

    convergence_info.corrected_dual_objective = corrected_dual_obj(buffer_kkt)

    gap = abs(convergence_info.primal_objective - convergence_info.dual_objective)
    abs_obj =
        abs(convergence_info.primal_objective) +
        abs(convergence_info.dual_objective)
    convergence_info.relative_optimality_gap = gap / (eps_ratio + abs_obj)

    convergence_info.candidate_type = candidate_type

    return convergence_info
end

"""
Compute infeasibility information of the given primal and dual solutions
"""
function compute_infeasibility_information(
    problem::CuLinearProgrammingProblem,
    primal_ray_estimate::CuVector{Float64},
    dual_ray_estimate::CuVector{Float64},
    candidate_type::PointType,
    primal_ray_estimate_product::CuVector{Float64},
    primal_ray_estimate_gradient::CuVector{Float64},
    buffer_kkt_infeas::BufferKKTState,
    buffer_lp::CuLinearProgrammingProblem,
)
    infeas_info = InfeasibilityInformation()

    primal_ray_inf_norm = CUDA.norm(primal_ray_estimate, Inf)
    if !iszero(primal_ray_inf_norm)
        primal_ray_estimate /= primal_ray_inf_norm
        primal_ray_estimate_product /= primal_ray_inf_norm
    end

    buffer_kkt_infeas.primal_solution = primal_ray_estimate
    buffer_kkt_infeas.primal_product = primal_ray_estimate_product

    buffer_lp.variable_lower_bound .= -1 ./ problem.isfinite_variable_lower_bound .+ 1
    buffer_lp.variable_upper_bound .= 1 ./ problem.isfinite_variable_upper_bound .- 1
    buffer_lp.objective_vector .= copy(problem.objective_vector)
    buffer_lp.right_hand_side .= 0.0

    compute_primal_residual!(buffer_lp, buffer_kkt_infeas)
    infeas_info.max_primal_ray_infeasibility = CUDA.norm([buffer_kkt_infeas.constraint_violation; buffer_kkt_infeas.lower_variable_violation; buffer_kkt_infeas.upper_variable_violation], Inf)
    infeas_info.primal_ray_linear_objective =
        CUDA.dot(problem.objective_vector, buffer_kkt_infeas.primal_solution)
    

    buffer_lp.variable_lower_bound .= copy(problem.variable_lower_bound)
    buffer_lp.variable_upper_bound .= copy(problem.variable_upper_bound)
    buffer_lp.objective_vector .= 0.0
    buffer_lp.right_hand_side .= copy(problem.right_hand_side)

    buffer_kkt_infeas.dual_solution = dual_ray_estimate
    buffer_kkt_infeas.primal_gradient .= primal_ray_estimate_gradient .- problem.objective_vector

    compute_dual_stats!(buffer_lp, buffer_kkt_infeas)

    scaling_factor = max(
        CUDA.norm(buffer_kkt_infeas.dual_solution, Inf),
        CUDA.norm(buffer_kkt_infeas.dual_stats.reduced_costs, Inf),
    )
    if !iszero(scaling_factor)
        infeas_info.max_dual_ray_infeasibility = buffer_kkt_infeas.dual_res_inf / scaling_factor
        infeas_info.dual_ray_objective =
        buffer_kkt_infeas.dual_stats.dual_objective / scaling_factor
    else
        infeas_info.max_dual_ray_infeasibility = 0.0
        infeas_info.dual_ray_objective = 0.0
    end

    infeas_info.candidate_type = candidate_type

    return infeas_info
end

"""
Compute iteration stats of the given primal and dual solutions
"""
function compute_iteration_stats(
    problem::CuLinearProgrammingProblem,
    qp_cache::CachedQuadraticProgramInfo,
    primal_iterate::CuVector{Float64},
    dual_iterate::CuVector{Float64},
    primal_ray_estimate::CuVector{Float64},
    dual_ray_estimate::CuVector{Float64},
    iteration_number::Integer,
    cumulative_kkt_matrix_passes::Float64,
    cumulative_time_sec::Float64,
    eps_optimal_absolute::Float64,
    eps_optimal_relative::Float64,
    step_size::Float64,
    primal_weight::Float64,
    candidate_type::PointType,
    primal_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
    primal_ray_estimate_product::CuVector{Float64},
    primal_ray_estimate_gradient::CuVector{Float64},
    buffer_kkt::BufferKKTState,
    buffer_kkt_infeas::BufferKKTState,
    buffer_lp::CuLinearProgrammingProblem,
)
    stats = IterationStats()
    stats.iteration_number = iteration_number
    stats.cumulative_kkt_matrix_passes = cumulative_kkt_matrix_passes
    stats.cumulative_time_sec = cumulative_time_sec

    stats.convergence_information = [
        compute_convergence_information(
        problem,
        qp_cache,
        primal_iterate,
        dual_iterate,
        eps_optimal_absolute / eps_optimal_relative,
        candidate_type,
        primal_product,
        primal_gradient,
        buffer_kkt,
        ),
    ]
    stats.infeasibility_information = [
        compute_infeasibility_information(
        problem,
        primal_ray_estimate,
        dual_ray_estimate,
        candidate_type,
        primal_ray_estimate_product,
        primal_ray_estimate_gradient,
        buffer_kkt_infeas,
        buffer_lp,
        ),
    ]
    stats.step_size = step_size
    stats.primal_weight = primal_weight
    stats.method_specific_stats = Dict{AbstractString,Float64}()

    return stats
end

mutable struct BufferOriginalSol
    original_primal_solution::CuVector{Float64}
    original_dual_solution::CuVector{Float64}
    original_primal_product::CuVector{Float64}
    original_primal_gradient::CuVector{Float64}
end

"""
Compute the iteration stats of the unscaled primal and dual solutions
"""
function evaluate_unscaled_iteration_stats(
    scaled_problem::CuScaledQpProblem,
    qp_cache::CachedQuadraticProgramInfo,
    termination_criteria::TerminationCriteria,
    record_iteration_stats::Bool,
    primal_solution::CuVector{Float64},
    dual_solution::CuVector{Float64},
    iteration::Int64,
    cumulative_time::Float64,
    cumulative_kkt_passes::Float64,
    eps_optimal_absolute::Float64,
    eps_optimal_relative::Float64,
    step_size::Float64,
    primal_weight::Float64,
    candidate_type::PointType,
    primal_product::CuVector{Float64},
    primal_gradient::CuVector{Float64},
    buffer_original::BufferOriginalSol,
    buffer_kkt::BufferKKTState,
    buffer_kkt_infeas::BufferKKTState,
    buffer_lp::CuLinearProgrammingProblem,
)
    # Unscale iterates.
    buffer_original.original_primal_solution .=
        primal_solution ./ scaled_problem.variable_rescaling
    buffer_original.original_primal_gradient .=
        primal_gradient .* scaled_problem.variable_rescaling
    buffer_original.original_dual_solution .=
        dual_solution ./ scaled_problem.constraint_rescaling
    buffer_original.original_primal_product .=
        primal_product .* scaled_problem.constraint_rescaling

    return compute_iteration_stats(
        scaled_problem.original_qp,
        qp_cache,
        buffer_original.original_primal_solution,
        buffer_original.original_dual_solution,
        buffer_original.original_primal_solution,  # ray estimate
        buffer_original.original_dual_solution,  # ray estimate
        iteration - 1,
        cumulative_kkt_passes,
        cumulative_time,
        eps_optimal_absolute,
        eps_optimal_relative,
        step_size,
        primal_weight,
        candidate_type,
        buffer_original.original_primal_product,
        buffer_original.original_primal_gradient,
        buffer_original.original_primal_product,
        buffer_original.original_primal_gradient,
        buffer_kkt,
        buffer_kkt_infeas,
        buffer_lp,
    )
end

#############################
# Below are print functions #
#############################
function print_to_screen_this_iteration(
    termination_reason::Union{TerminationReason,Bool},
    iteration::Int64,
    verbosity::Int64,
    termination_evaluation_frequency::Int32,
)
    if verbosity >= 2
        if termination_reason == false
        num_of_evaluations = (iteration - 1) / termination_evaluation_frequency
        if verbosity >= 9
            display_frequency = 1
        elseif verbosity >= 6
            display_frequency = 3
        elseif verbosity >= 5
            display_frequency = 10
        elseif verbosity >= 4
            display_frequency = 20
        elseif verbosity >= 3
            display_frequency = 50
        else
            return iteration == 1
        end
        # print_to_screen_this_iteration is true every
        # display_frequency * termination_evaluation_frequency iterations.
        return mod(num_of_evaluations, display_frequency) == 0
        else
        return true
        end
    else
        return false
    end
end


function display_iteration_stats_heading(show_infeasibility::Bool)
    Printf.@printf(
        "%s | %s | %s | %s |",
        rpad("runtime", 24),
        rpad("residuals", 26),
        rpad(" solution information", 26),
        rpad("relative residuals", 23)
    )
    if show_infeasibility
        Printf.@printf(" %s | %s |", rpad("primal ray", 27), rpad("dual ray", 18))
    end
    println("")
    Printf.@printf(
        "%s %s %s | %s %s  %s | %s %s %s | %s %s %s |",
        rpad("#iter", 7),
        rpad("#kkt", 8),
        rpad("seconds", 7),
        rpad("pr norm", 8),
        rpad("du norm", 8),
        rpad("gap", 7),
        rpad(" pr obj", 9),
        rpad("pr norm", 8),
        rpad("du norm", 7),
        rpad("rel pr", 7),
        rpad("rel du", 7),
        rpad("rel gap", 7)
    )
    if show_infeasibility
        Printf.@printf(
        # " %s %s %s | %s %s |",
        " %s %s | %s %s |",
        rpad("pr norm", 9),
        rpad("linear", 8),
        # rpad("qu norm", 8),
        rpad("du norm", 9),
        rpad("dual obj", 8)
        )
    end
    print("\n")
end


function display_iteration_stats_heading(verbosity::Int64)
    if verbosity >= 7
        display_iteration_stats_heading(true)
    elseif verbosity >= 2
        display_iteration_stats_heading(false)
    end
end

function lpad_float(number::Float64)
    return lpad(Printf.@sprintf("%.1e", number), 8)
end

function display_iteration_stats(
    stats::IterationStats,
    show_infeasibility::Bool,
)
    if length(stats.convergence_information) > 0
        Printf.@printf(
        "%s  %.1e  %.1e | %.1e  %.1e  %s | %s  %.1e  %.1e | %.1e %.1e %.1e |",
        rpad(string(stats.iteration_number), 6),
        stats.cumulative_kkt_matrix_passes,
        stats.cumulative_time_sec,
        stats.convergence_information[1].l2_primal_residual,
        stats.convergence_information[1].l2_dual_residual,
        lpad_float(
            stats.convergence_information[1].primal_objective -
            stats.convergence_information[1].dual_objective,
        ),
        lpad_float(stats.convergence_information[1].primal_objective),
        stats.convergence_information[1].l2_primal_variable,
        stats.convergence_information[1].l2_dual_variable,
        stats.convergence_information[1].relative_l2_primal_residual,
        stats.convergence_information[1].relative_l2_dual_residual,
        stats.convergence_information[1].relative_optimality_gap
        )
    else
        Printf.@printf(
        "%s  %.1e  %.1e",
        rpad(string(stats.iteration_number), 6),
        stats.cumulative_kkt_matrix_passes,
        stats.cumulative_time_sec
        )
    end

    if show_infeasibility
        if length(stats.infeasibility_information) > 0
        Printf.@printf(
            " %.1e  %s  | %.1e  %s  |",
            # " %.1e  %s  %.1e  | %.1e  %s  |",
            stats.infeasibility_information[1].max_primal_ray_infeasibility,
            lpad_float(
            stats.infeasibility_information[1].primal_ray_linear_objective,
            ),
            stats.infeasibility_information[1].max_dual_ray_infeasibility,
            lpad_float(stats.infeasibility_information[1].dual_ray_objective)
        )
        end
    end

    print("\n")
end

function display_iteration_stats(stats::IterationStats, verbosity::Int64)
    if verbosity >= 7
        display_iteration_stats(stats, true)
    else
        display_iteration_stats(stats, false)
    end
end

function print_infinity_norms(convergence_info::ConvergenceInformation)
    print("l_inf: ")
    Printf.@printf(
        "primal_res = %.3e, dual_res = %.3e, primal_var = %.3e, dual_var = %.3e",
        convergence_info.l_inf_primal_residual,
        convergence_info.l_inf_dual_residual,
        convergence_info.l_inf_primal_variable,
        convergence_info.l_inf_dual_variable
    )
    println()
end    