using Test
import MathOptInterface as MOI
import cuPDLP
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
atol = 1e-2,
rtol = 1e-2,
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
include = [
    r"^test_conic_linear_VectorAffineFunction$",
    r"^test_conic_linear_VectorAffineFunction_2$",
    r"^test_conic_linear_VectorOfVariables$",
    r"^test_conic_linear_VectorOfVariables_2$",
    r"^test_constraint_ScalarAffineFunction_LessThan$",
    r"^test_constraint_ScalarAffineFunction_duplicate$",
    r"^test_constraint_VectorAffineFunction_duplicate$",
    r"^test_infeasible_MAX_SENSE$",
    r"^test_infeasible_MAX_SENSE_offset$",
    r"^test_infeasible_MIN_SENSE$",
    r"^test_infeasible_MIN_SENSE_offset$",
    r"^test_infeasible_affine_MAX_SENSE$",
    r"^test_infeasible_affine_MAX_SENSE_offset$",
    r"^test_infeasible_affine_MIN_SENSE$",
    r"^test_infeasible_affine_MIN_SENSE_offset$",
    r"^test_linear_INFEASIBLE$",
    r"^test_linear_INFEASIBLE_2$",
    r"^test_linear_Interval_inactive$",
    r"^test_linear_integration$",
    r"^test_linear_integration_2$",
    r"^test_linear_integration_delete_variables$",
    r"^test_linear_variable_open_intervals$",
    r"^test_modification_coef_scalar_objective$",
    r"^test_modification_coef_scalaraffine_lessthan$",
    r"^test_modification_delete_variable_with_single_variable_obj$",
    r"^test_modification_delete_variables_in_a_batch$",
    r"^test_modification_func_scalaraffine_lessthan$",
    r"^test_modification_set_scalaraffine_lessthan$",
    r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_lower$",
    r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_upper$",
    r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_GreaterThan$",
    r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_lower$",
    r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_upper$",
    r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_LessThan$",
    r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan$",
    r"^test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan_max$",
    r"^test_unbounded_MAX_SENSE_offset$",
    r"^test_unbounded_MIN_SENSE_offset$",
]
)

