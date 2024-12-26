#include <gtest/gtest.h>

#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

namespace malyshev_v_conjugate_gradient_method {

void run_conjugate_gradient_test(int size, std::vector<double> matrix, std::vector<double> rhs,
                                 std::vector<double> initial_guess, std::vector<double> expected,
                                 double tolerance = 1e-6) {
  std::vector<double> result(expected.size());
  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&tolerance));
  task->inputs_count.emplace_back(1);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  task->inputs_count.emplace_back(matrix.size());
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
  task->inputs_count.emplace_back(rhs.size());
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_guess.data()));
  task->inputs_count.emplace_back(initial_guess.size());
  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  ConjugateGradientSolver solver(task);
  ASSERT_TRUE(solver.validation());
  solver.pre_processing();
  solver.run();
  solver.post_processing();
  for (size_t i = 0; i < expected.size(); i++) ASSERT_NEAR(result[i], expected[i], tolerance);
}

}  // namespace malyshev_v_conjugate_gradient_method

TEST(malyshev_v_conjugate_gradient_method, test_small_system) {
  int size = 2;
  std::vector<double> matrix = {5, 2, 2, 3};
  std::vector<double> rhs = {1, 2};
  std::vector<double> initial_guess = {0, 0};
  std::vector<double> expected = {0.09090909, 0.63636364};
  double tolerance = 1e-6;
  malyshev_v_conjugate_gradient_method::run_conjugate_gradient_test(size, matrix, rhs, initial_guess, expected,
                                                                    tolerance);
}

TEST(malyshev_v_conjugate_gradient_method, test_medium_system) {
  int size = 3;
  std::vector<double> matrix = {10, 2, 1, 2, 10, 3, 1, 3, 10};
  std::vector<double> rhs = {1, 2, 3};
  std::vector<double> initial_guess = {0, 0, 0};
  std::vector<double> expected = {0.08080808, 0.16161616, 0.28282828};
  double tolerance = 1e-6;
  malyshev_v_conjugate_gradient_method::run_conjugate_gradient_test(size, matrix, rhs, initial_guess, expected,
                                                                    tolerance);
}

TEST(malyshev_v_conjugate_gradient_method, test_large_system) {
  int size = 4;
  std::vector<double> matrix = {4, 1, 0, 0, 1, 4, 1, 0, 0, 1, 4, 1, 0, 0, 1, 4};
  std::vector<double> rhs = {1, 2, 3, 4};
  std::vector<double> initial_guess = {0, 0, 0, 0};
  std::vector<double> expected = {0.09090909, 0.18181818, 0.27272727, 0.36363636};
  double tolerance = 1e-6;
  malyshev_v_conjugate_gradient_method::run_conjugate_gradient_test(size, matrix, rhs, initial_guess, expected,
                                                                    tolerance);
}