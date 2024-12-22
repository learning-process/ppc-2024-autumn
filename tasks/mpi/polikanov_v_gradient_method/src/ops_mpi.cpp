#include "mpi/polikanov_v_gradient_method/include/ops_mpi.hpp"

namespace polikanov_v_gradient_method_mpi {

bool GradientMethod::validation() {
  internal_order_test();
  return true;
}

bool GradientMethod::pre_processing() {
  internal_order_test();
  size = *reinterpret_cast<int*>(taskData->inputs[0]);
  tolerance = *reinterpret_cast<double*>(taskData->inputs[1]);
  auto* flat_matrix_ptr = reinterpret_cast<double*>(taskData->inputs[2]);
  int flat_matrix_size = taskData->inputs_count[2];
  flat_matrix.assign(flat_matrix_ptr, flat_matrix_ptr + flat_matrix_size);
  auto* rhs_ptr = reinterpret_cast<double*>(taskData->inputs[3]);
  int rhs_size = taskData->inputs_count[3];
  rhs.assign(rhs_ptr, rhs_ptr + rhs_size);
  auto* initial_guess_ptr = reinterpret_cast<double*>(taskData->inputs[4]);
  int initial_guess_size = taskData->inputs_count[4];
  initial_guess.assign(initial_guess_ptr, initial_guess_ptr + initial_guess_size);
  result.resize(size);
  return true;
}

bool GradientMethod::run() {
  internal_order_test();
  result = ConjugateGradientMethod(world, flat_matrix, rhs, initial_guess, tolerance, size);
  return true;
}

bool GradientMethod::post_processing() {
  internal_order_test();
  auto* result_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), result_ptr);
  return true;
}

}  // namespace polikanov_v_gradient_method_mpi