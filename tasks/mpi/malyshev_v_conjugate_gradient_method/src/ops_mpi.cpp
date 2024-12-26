#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_method {

bool ConjugateGradientSolver::validation() {
  internal_order_test();
  return true;
}

bool ConjugateGradientSolver::pre_processing() {
  internal_order_test();
  size = *reinterpret_cast<int*>(taskData->inputs[0]);
  tolerance = *reinterpret_cast<double*>(taskData->inputs[1]);
  auto* matrix_ptr = reinterpret_cast<double*>(taskData->inputs[2]);
  int matrix_size = taskData->inputs_count[2];
  matrix.assign(matrix_ptr, matrix_ptr + matrix_size);
  auto* rhs_ptr = reinterpret_cast<double*>(taskData->inputs[3]);
  int rhs_size = taskData->inputs_count[3];
  rhs.assign(rhs_ptr, rhs_ptr + rhs_size);
  auto* initial_guess_ptr = reinterpret_cast<double*>(taskData->inputs[4]);
  int initial_guess_size = taskData->inputs_count[4];
  initial_guess.assign(initial_guess_ptr, initial_guess_ptr + initial_guess_size);
  result.resize(size);
  return true;
}

bool ConjugateGradientSolver::run() {
  internal_order_test();
  result = SolveConjugateGradient(world, matrix, rhs, initial_guess, tolerance, size);
  return true;
}

bool ConjugateGradientSolver::post_processing() {
  internal_order_test();
  auto* result_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}

}  // namespace malyshev_v_conjugate_gradient_method