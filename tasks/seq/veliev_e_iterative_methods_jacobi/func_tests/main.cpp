#include <gtest/gtest.h>

#include <vector>

#include "seq/veliev_e_iterative_methods_jacobi/include/ops_seq.hpp"

TEST(veliev_iter_methods, Identity_Matrix_Test) {
  const int size = 3;

  std::vector<double> matrixA = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> vectorB = {5, -3, 2};
  std::vector<double> solution(size, 0.0);

  auto dataPack = std::make_shared<ppc::core::TaskData>();
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(vectorB.data()));
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  dataPack->inputs_count.emplace_back(size);
  dataPack->inputs_count.emplace_back(vectorB.size());
  dataPack->inputs_count.emplace_back(solution.size());
  dataPack->outputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  dataPack->outputs_count.emplace_back(solution.size());

  veliev_e_iterative_methods_jacobi::MethodJacobi solver(dataPack);

  ASSERT_EQ(solver.validation(), true);
  solver.pre_processing();
  solver.run();
  solver.post_processing();

  ASSERT_NEAR(solution[0], 5.0, 1e-4);
  ASSERT_NEAR(solution[1], -3.0, 1e-4);
  ASSERT_NEAR(solution[2], 2.0, 1e-4);
}

TEST(veliev_iter_methods, Diagonal_Matrix_Test) {
  const int size = 3;

  std::vector<double> matrixA = {3, 0, 0, 0, 5, 0, 0, 0, 7};
  std::vector<double> vectorB = {9, 25, 49};
  std::vector<double> solution(size, 0.0);

  auto dataPack = std::make_shared<ppc::core::TaskData>();
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(vectorB.data()));
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  dataPack->inputs_count.emplace_back(size);
  dataPack->inputs_count.emplace_back(vectorB.size());
  dataPack->inputs_count.emplace_back(solution.size());
  dataPack->outputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  dataPack->outputs_count.emplace_back(solution.size());

  veliev_e_iterative_methods_jacobi::MethodJacobi solver(dataPack);

  ASSERT_EQ(solver.validation(), true);
  solver.pre_processing();
  solver.run();
  solver.post_processing();

  ASSERT_NEAR(solution[0], 3.0, 1e-4);
  ASSERT_NEAR(solution[1], 5.0, 1e-4);
  ASSERT_NEAR(solution[2], 7.0, 1e-4);
}

TEST(veliev_iter_methods, Invalid_Matrix_With_Zero_Diagonal_Element) {
  const int size = 2;

  std::vector<double> matrixA = {0, 1, 1, 3};
  std::vector<double> vectorB = {1, 2};
  std::vector<double> solution(size, 0.0);

  auto dataPack = std::make_shared<ppc::core::TaskData>();
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA.data()));
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(vectorB.data()));
  dataPack->inputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  dataPack->inputs_count.emplace_back(size);
  dataPack->inputs_count.emplace_back(vectorB.size());
  dataPack->inputs_count.emplace_back(solution.size());
  dataPack->outputs.emplace_back(reinterpret_cast<uint8_t *>(solution.data()));
  dataPack->outputs_count.emplace_back(solution.size());

  veliev_e_iterative_methods_jacobi::MethodJacobi solver(dataPack);

  ASSERT_EQ(solver.validation(), true);
  ASSERT_EQ(solver.pre_processing(), false);
}
