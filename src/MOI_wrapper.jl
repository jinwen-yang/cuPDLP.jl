import MathOptInterface as MOI

# Inspired from `Clp.jl/src/MOI_wrapper/MOI_wrapper.jl`
MOI.Utilities.@product_of_sets(
    _LPProductOfSets,
    MOI.EqualTo{T},
    MOI.LessThan{T},
    MOI.GreaterThan{T},
    MOI.Interval{T},
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

const SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

"""
    Optimizer()

Create a new cuPDLP optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    output::Union{Nothing,SaddlePointOutput}
    max_sense::Bool
    silent::Bool
    parameters::PdhgParameters

    function Optimizer()
        return new(
            nothing,
            false,
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
    ::Type{<:Union{MOI.VariableIndex,MOI.ScalarAffineFunction{Float64}}},
    ::Type{<:SCALAR_SETS},
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
    problem = TwoSidedQpProblem(
        src.variables.lower,
        src.variables.upper,
        row_bounds.lower,
        row_bounds.upper,
        A,
        MOI.constant(obj),
        dest.max_sense ? -c : c,
        spzeros(A.n, A.n),
    )
    dest.output = optimize(dest.parameters, transform_to_standard_form(problem))
    return MOI.Utilities.identity_index_map(src), false
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    MOI.optimize!(dest, cache)
    return index_map, false
end
