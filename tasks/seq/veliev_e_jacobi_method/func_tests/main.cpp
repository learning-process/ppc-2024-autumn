// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/veliev_e_jacobi_method/include/ops_seq.hpp"

TEST(veliev_e_jacobi_method, Test_2x2_System) {
  const uint32_t systemSize = 2;

  std::vector<double> matrixData = {4, 1, 1, 3};
  std::vector<double> rhsVector = {1, 2};
  std::vector<double> solutionVector(systemSize, 0.0);

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrixData.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhsVector.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(solutionVector.data()));
  taskDataContainer->inputs_count.push_back(systemSize);
  taskDataContainer->inputs_count.push_back(rhsVector.size());
  taskDataContainer->inputs_count.push_back(solutionVector.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(solutionVector.data()));
  taskDataContainer->outputs_count.push_back(solutionVector.size());

  veliev_e_jacobi_method::MethodJacobi jacobiSolver(taskDataContainer);

  ASSERT_TRUE(jacobiSolver.validation());

  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(solutionVector[0], 0.1, 3e-2);
  ASSERT_NEAR(solutionVector[1], 0.6, 5e-2);
}

TEST(veliev_e_jacobi_method, Test_3x3_System) {
  const uint32_t systemDim = 3;

  std::vector<double> matrixData = {4, 1, 1, 1, 3, 1, 1, 1, 3};
  std::vector<double> rhsVector = {3, 2.5, 2.5};
  std::vector<double> solutionVector(systemDim, 0.0);

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrixData.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhsVector.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(solutionVector.data()));
  taskDataContainer->inputs_count.push_back(systemDim);
  taskDataContainer->inputs_count.push_back(rhsVector.size());
  taskDataContainer->inputs_count.push_back(solutionVector.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(solutionVector.data()));
  taskDataContainer->outputs_count.push_back(solutionVector.size());

  veliev_e_jacobi_method::MethodJacobi jacobiSolver(taskDataContainer);

  ASSERT_TRUE(jacobiSolver.validation());

  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(solutionVector[0], 0.5, 1e-4);
  ASSERT_NEAR(solutionVector[1], 0.5, 1e-4);
  ASSERT_NEAR(solutionVector[2], 0.5, 1e-4);
}

TEST(veliev_e_jacobi_method, Test_Diagonal_Dominance) {
  const uint32_t matrixSize = 3;

  std::vector<double> matrix = {4, 2, 0, 1, 5, 1, 0, 2, 4};
  std::vector<double> rhs(matrixSize, 0.0);
  rhs[0] = 1;
  rhs[1] = 2.2;
  rhs[2] = 3;
  std::vector<double> solution(matrixSize, 0.0);

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->inputs_count.push_back(matrixSize);
  taskDataContainer->inputs_count.push_back(rhs.size());
  taskDataContainer->inputs_count.push_back(solution.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->outputs_count.push_back(solution.size());

  veliev_e_jacobi_method::MethodJacobi jacobiSolver(taskDataContainer);

  ASSERT_TRUE(jacobiSolver.validation());

  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(solution[0], 0.1, 1e-6);
  ASSERT_NEAR(solution[1], 0.3, 1e-6);
  ASSERT_NEAR(solution[2], 0.6, 1e-6);
}

TEST(veliev_e_jacobi_method, Test_empty_System) {
  const uint32_t systemSize = 0;

  std::vector<double> emptyMatrix;
  std::vector<double> emptyRhs;
  std::vector<double> emptySolution(systemSize, 0.0);

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(emptyMatrix.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(emptyRhs.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(emptySolution.data()));
  taskDataContainer->inputs_count.push_back(systemSize);
  taskDataContainer->inputs_count.push_back(emptyRhs.size());
  taskDataContainer->inputs_count.push_back(emptySolution.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(emptySolution.data()));
  taskDataContainer->outputs_count.push_back(emptySolution.size());

  veliev_e_jacobi_method::MethodJacobi jacobiProcessor(taskDataContainer);

  ASSERT_FALSE(jacobiProcessor.validation());
}

TEST(veliev_e_jacobi_method, Test_zero_diagonal_element) {
  const uint32_t systemSize = 3;

  std::vector<double> matrix = {4, 2, 0, 1, 0, 1, 0, 2, 4};
  std::vector<double> rhs = {1, 2.2, 3};
  std::vector<double> solution(systemSize, 0.0);

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->inputs_count.push_back(systemSize);
  taskDataContainer->inputs_count.push_back(rhs.size());
  taskDataContainer->inputs_count.push_back(solution.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->outputs_count.push_back(solution.size());

  veliev_e_jacobi_method::MethodJacobi jacobiSolver(taskDataContainer);

  ASSERT_TRUE(jacobiSolver.validation());
  ASSERT_FALSE(jacobiSolver.pre_processing());
}

TEST(veliev_e_jacobi_method, Test_1x1_System) {
  const uint32_t singleElementSystem = 1;

  std::vector<double> matrix = {1};
  std::vector<double> rhs = {5};
  std::vector<double> solution(singleElementSystem, 0.0);

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->inputs_count.push_back(singleElementSystem);
  taskDataContainer->inputs_count.push_back(rhs.size());
  taskDataContainer->inputs_count.push_back(solution.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->outputs_count.push_back(solution.size());

  veliev_e_jacobi_method::MethodJacobi jacobiSolver(taskDataContainer);

  ASSERT_TRUE(jacobiSolver.validation());

  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(solution[0], 5.0, 1e-6);
}

TEST(veliev_e_jacobi_method, Test_Zero_Right_Side) {
  const uint32_t systemSize = 3;

  std::vector<double> matrix = {4, 1, 1, 1, 3, 1, 1, 1, 3};
  std::vector<double> rhs = {0, 0, 0};
  std::vector<double> solution(systemSize, 0.0);

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->inputs_count.push_back(systemSize);
  taskDataContainer->inputs_count.push_back(rhs.size());
  taskDataContainer->inputs_count.push_back(solution.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->outputs_count.push_back(solution.size());

  veliev_e_jacobi_method::MethodJacobi jacobiSolver(taskDataContainer);

  ASSERT_TRUE(jacobiSolver.validation());
  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_EQ(solution[0], 0.0);
  ASSERT_EQ(solution[1], 0.0);
  ASSERT_EQ(solution[2], 0.0);
}

TEST(veliev_e_jacobi_method, Test_Diagonal_Dominance_Large_Error) {
  const uint32_t systemSize = 3;

  std::vector<double> matrix = {1000, 1, 1, 1, 1000, 1, 1, 1, 1000};
  std::vector<double> rhs = {3, 2.5, 2.5};
  std::vector<double> solution(systemSize, 0.0);

  auto taskDataContainer = std::make_shared<ppc::core::TaskData>();
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
  taskDataContainer->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->inputs_count.push_back(systemSize);
  taskDataContainer->inputs_count.push_back(rhs.size());
  taskDataContainer->inputs_count.push_back(solution.size());
  taskDataContainer->outputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
  taskDataContainer->outputs_count.push_back(solution.size());

  veliev_e_jacobi_method::MethodJacobi jacobiSolver(taskDataContainer);

  ASSERT_TRUE(jacobiSolver.validation());
  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(solution[0], 0.003, 1e-4);
  ASSERT_NEAR(solution[1], 0.0025, 1e-4);
  ASSERT_NEAR(solution[2], 0.0025, 1e-4);
}
