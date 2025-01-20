module TestMOI

using Test
import MathOptInterface as MOI
import cuPDLP

function test_runtests()
    optimizer = cuPDLP.Optimizer()
    MOI.set(optimizer, MOI.Silent(), true) # comment this to enable output
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            optimizer,
        ),
        Float64,
    )
    config = MOI.Test.Config(
        rtol = 1e-2,
        atol = 1e-2,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
            MOI.SolverVersion,
        ],
    )
    MOI.Test.runtests(
        model,
	config,
	exclude = [
	    # No constraint so cuPDLP fails with the CUDA error
	    # `Grid dimensions should be non-null`
            r"^test_variable_solve_with_lowerbound$",
            r"^test_variable_solve_with_upperbound$",
            r"^test_solve_result_index$",
            r"^test_solve_TerminationStatus_DUAL_INFEASIBLE$",
            r"^test_modification_transform_singlevariable_lessthan$",
            r"^test_modification_const_scalar_objective$",
            r"^test_DualObjectiveValue_Max_VariableIndex_LessThan$",
            r"^test_DualObjectiveValue_Min_VariableIndex_GreaterThan$",
            r"^test_attribute_RawStatusString$",
            r"^test_attribute_SolveTimeSec$",
            r"^test_solve_optimize_twice$",
            r"^test_solve_VariableIndex_ConstraintDual_MAX_SENSE$",
            r"^test_solve_VariableIndex_ConstraintDual_MIN_SENSE$",
            r"^test_modification_set_singlevariable_lessthan$",
            r"^test_objective_ObjectiveFunction_VariableIndex$",
            r"^test_objective_ObjectiveFunction_blank$",
            r"^test_objective_ObjectiveFunction_constant$",
            r"^test_objective_ObjectiveFunction_duplicate_terms$",
            r"^test_objective_FEASIBILITY_SENSE_clears_objective$",
            r"^test_modification_coef_scalar_objective$",
            r"^test_linear_variable_open_intervals$",
            r"^test_modification_delete_variable_with_single_variable_obj$",
            r"^test_modification_delete_variables_in_a_batch$",
            # Not all constraints have finite bounds on at least one side.
            r"^test_linear_open_intervals$",
        ]
    )
    return
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

end  # module

TestMOI.runtests()
