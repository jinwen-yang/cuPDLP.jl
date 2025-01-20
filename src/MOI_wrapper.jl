import MathOptInterface as MOI

# Inspired from `Clp.jl/src/MOI_wrapper/MOI_wrapper.jl`
MOI.Utilities.@product_of_sets(
    _LPProductOfSets,
    MOI.EqualTo{T},
    MOI.GreaterThan{T},
)

const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int64,
            MOI.Utilities.OneBasedIndexing,
        },
        MOI.Utilities.Hyperrectangle{Float64},
        _LPProductOfSets{Float64},
    },
}

const DEFAULT_OPTIONS = Dict{String,Any}(
    "max_iters" => 100,
    "ϵ_primal" => 1e-4,
    "ϵ_dual" => 1e-4,
    "ϵ_gap" => 1e-4,
    "ϵ_unbounded" => 1e-7,
    "ϵ_infeasible" => 1e-7,
)

Base.show(io::IO, ::Type{OptimizerCache}) = print(io, "cuPDLP.OptimizerCache")

const BOUND_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const ROW_SETS = Union{
    MOI.EqualTo{Float64},
    MOI.GreaterThan{Float64},
}

"""
    Optimizer()

Create a new cuPDLP optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    output::Union{Nothing,SaddlePointOutput}
    max_sense::Bool
    num_equalities::Int
    silent::Bool
    parameters::PdhgParameters

    function Optimizer()
        return new(
            nothing,
            false,
	    0,
            false,
            PdhgParameters(
                10,
                false,
                1.0,
                1.0,
                true,
                2,
                true,
                64,
                construct_termination_criteria(),
                construct_restart_parameters(
                    cuPDLP.ADAPTIVE_KKT,    # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE_KKT
                    cuPDLP.KKT_GREEDY,      # NO_RESTART_TO_CURRENT KKT_GREEDY
                    1000,                   # restart_frequency_if_fixed
                    0.36,                   # artificial_restart_threshold
                    0.2,                    # sufficient_reduction_for_restart
                    0.8,                    # necessary_reduction_for_restart
                    0.5,                    # primal_weight_update_smoothing
                ),
                AdaptiveStepsizeParams(0.3, 0.6),
            ),
        )
    end
end

function MOI.default_cache(::Optimizer, ::Type)
    return OptimizerCache()
end

# ====================
#   empty functions
# ====================

function MOI.is_empty(optimizer::Optimizer)
    return isnothing(optimizer.output)
end

function MOI.empty!(optimizer::Optimizer)
    optimizer.output = nothing
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "cuPDLP"

# MOI.RawOptimizerAttribute

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return hasfield(PdhgParameters, Symbol(param.name))
end

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    setfield!(optimizer.parameters, Symbol(param.name), value)
    return
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    return getfield(optimizer.parameters, Symbol(param.name))
end

# MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

# ========================================
#   Supported constraints and objectives
# ========================================

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:BOUND_SETS},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{<:ROW_SETS},
)
    return true
end


MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

# ===============================
#   Optimize and post-optimize
# ===============================

function _flip_sense(optimizer::Optimizer, obj)
    return optimizer.max_sense ? -obj : obj
end

function MOI.optimize!(dest::Optimizer, src::OptimizerCache)
    MOI.empty!(dest)
    Ab = src.constraints
    A = convert(SparseMatrixCSC{Float64,Int64}, Ab.coefficients)
    row_bounds = src.constraints.constants
    dest.max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    c = zeros(A.n)
    for term in obj.terms
        c[term.variable.value] += term.coefficient
    end
    dest.num_equalities = MOI.get(src, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}())
    problem = QuadraticProgrammingProblem(
        src.variables.lower,
        src.variables.upper,
        spzeros(A.n, A.n),
        _flip_sense(dest, c),
        _flip_sense(dest, MOI.constant(obj)),
        A,
        row_bounds.lower,
	dest.num_equalities,
    )
    if dest.silent
        verbosity = dest.parameters.verbosity
        dest.parameters.verbosity = 0
    end
    dest.output = optimize(dest.parameters, problem)
    if dest.silent # restore it
        dest.parameters.verbosity = verbosity
    end
    return MOI.Utilities.identity_index_map(src), false
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    MOI.optimize!(dest, cache)
    return index_map, false
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.output.iteration_status[end].cumulative_time_sec
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    if isnothing(optimizer.output)
        return "Optimize not called"
    else
        return string(optimizer.output.termination_reason) * " : " * optimizer.output.termination_string
    end
end

const _TERMINATION_STATUS_MAP = Dict(
    TERMINATION_REASON_UNSPECIFIED => MOI.OPTIMIZE_NOT_CALLED,
    TERMINATION_REASON_OPTIMAL => MOI.OPTIMAL,
    TERMINATION_REASON_PRIMAL_INFEASIBLE => MOI.INFEASIBLE,
    TERMINATION_REASON_DUAL_INFEASIBLE => MOI.DUAL_INFEASIBLE,
    TERMINATION_REASON_TIME_LIMIT => MOI.TIME_LIMIT,
    TERMINATION_REASON_ITERATION_LIMIT => MOI.ITERATION_LIMIT,
    TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT => MOI.NUMERICAL_ERROR,
    TERMINATION_REASON_NUMERICAL_ERROR => MOI.NUMERICAL_ERROR,
    TERMINATION_REASON_INVALID_PROBLEM => MOI.INVALID_MODEL,
    TERMINATION_REASON_OTHER => MOI.OTHER_ERROR,
)

# Implements getter for result value and statuses
function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    return isnothing(optimizer.output) ? MOI.OPTIMIZE_NOT_CALLED :
           _TERMINATION_STATUS_MAP[optimizer.output.termination_reason]
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return _flip_sense(optimizer, optimizer.output.iteration_stats[end].convergence_information[].primal_objective)
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return _flip_sense(optimizer, optimizer.output.iteration_stats[end].convergence_information[].dual_objective)
end

const _PRIMAL_STATUS_MAP = Dict(
    TERMINATION_REASON_UNSPECIFIED => MOI.NO_SOLUTION,
    TERMINATION_REASON_OPTIMAL => MOI.FEASIBLE_POINT,
    TERMINATION_REASON_PRIMAL_INFEASIBLE => MOI.NO_SOLUTION,
    TERMINATION_REASON_DUAL_INFEASIBLE => MOI.INFEASIBILITY_CERTIFICATE,
    TERMINATION_REASON_TIME_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_ITERATION_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_NUMERICAL_ERROR => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_INVALID_PROBLEM => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_OTHER => MOI.UNKNOWN_RESULT_STATUS,
)

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return _PRIMAL_STATUS_MAP[optimizer.output.termination_reason]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.output.primal_solution[vi.value]
end

const _DUAL_STATUS_MAP = Dict(
    TERMINATION_REASON_UNSPECIFIED => MOI.NO_SOLUTION,
    TERMINATION_REASON_OPTIMAL => MOI.FEASIBLE_POINT,
    TERMINATION_REASON_PRIMAL_INFEASIBLE => MOI.INFEASIBILITY_CERTIFICATE,
    TERMINATION_REASON_DUAL_INFEASIBLE => MOI.NO_SOLUTION,
    TERMINATION_REASON_TIME_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_ITERATION_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_NUMERICAL_ERROR => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_INVALID_PROBLEM => MOI.UNKNOWN_RESULT_STATUS,
    TERMINATION_REASON_OTHER => MOI.UNKNOWN_RESULT_STATUS,
)

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    return _DUAL_STATUS_MAP[optimizer.output.termination_reason]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.output.dual_solution[ci.value]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.output.dual_solution[optimizer.num_equalities + ci.value]
end

function MOI.get(optimizer::Optimizer, ::MOI.ResultCount)
    if isnothing(optimizer.output)
        return 0
    else
        return 1
    end
end
