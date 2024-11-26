// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kozlova_e_jacobi_method/include/ops_seq.hpp"


TEST(kozlova_e_jacobi_method, Test_2x2_System) {
  const int N = 2;

  // 4x + y = 1
  // x + 3y = 2

  std::vector<double> A = {4, 1, 1, 3};
  std::vector<double> B = {1, 2};  
  std::vector<double> X(N, 0.0);   

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->inputs_count.emplace_back(N);
  taskData->inputs_count.emplace_back(B.size());
  taskData->inputs_count.emplace_back(X.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->outputs_count.emplace_back(X.size());

  
  kozlova_e_jacobi_method::MethodJacobi jacobiSolver(taskData);
  ASSERT_EQ(jacobiSolver.validation(), true);
  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(X[0], 0.1, 3e-2);
  ASSERT_NEAR(X[1], 0.6, 5e-2);
}

TEST(kozlova_e_jacobi_method, Test_3x3_System) {
  const int N = 3;

  // 4x + y + z = 3
  // x + 3y + z = 2.5
  // x + y + 3z = 2.5

  std::vector<double> A = {4, 1, 1, 1, 3, 1, 1, 1, 3};
  std::vector<double> B = {3, 2.5, 2.5};
  std::vector<double> X(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->inputs_count.emplace_back(N);
  taskData->inputs_count.emplace_back(B.size());
  taskData->inputs_count.emplace_back(X.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->outputs_count.emplace_back(X.size());

  kozlova_e_jacobi_method::MethodJacobi jacobiSolver(taskData);
  ASSERT_EQ(jacobiSolver.validation(), true);
  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(X[0], 0.5, 1e-4);
  ASSERT_NEAR(X[1], 0.5, 1e-4);
  ASSERT_NEAR(X[2], 0.5, 1e-4);
}

TEST(kozlova_e_jacobi_method, Test_Diagonal_Dominance) {
  const int N = 3;

  std::vector<double> A = {4, 2, 0, 1, 5, 1, 0, 2, 4};
  std::vector<double> B = {1, 2.2, 3};
  std::vector<double> X(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->inputs_count.emplace_back(N);
  taskData->inputs_count.emplace_back(B.size());
  taskData->inputs_count.emplace_back(X.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->outputs_count.emplace_back(X.size());

  kozlova_e_jacobi_method::MethodJacobi jacobiSolver(taskData);
  ASSERT_EQ(jacobiSolver.validation(), true);
  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(X[0], 0.1, 1e-6);
  ASSERT_NEAR(X[1], 0.3, 1e-6);
  ASSERT_NEAR(X[2], 0.6, 1e-6);
}

TEST(kozlova_e_jacobi_method, Test_empty_System) {
  const int N = 0;

  std::vector<double> A = {};
  std::vector<double> B = {};
  std::vector<double> X(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->inputs_count.emplace_back(N);
  taskData->inputs_count.emplace_back(B.size());
  taskData->inputs_count.emplace_back(X.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->outputs_count.emplace_back(X.size());

  kozlova_e_jacobi_method::MethodJacobi jacobiSolver(taskData);
  ASSERT_EQ(jacobiSolver.validation(), false);
}

TEST(kozlova_e_jacobi_method, Test_zero_diagonal_element) {
  const int N = 3;

  std::vector<double> A = {4, 2, 0, 1, 0, 1, 0, 2, 4};
  std::vector<double> B = {1, 2.2, 3};
  std::vector<double> X(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->inputs_count.emplace_back(N);
  taskData->inputs_count.emplace_back(B.size());
  taskData->inputs_count.emplace_back(X.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->outputs_count.emplace_back(X.size());

  kozlova_e_jacobi_method::MethodJacobi jacobiSolver(taskData);
  ASSERT_EQ(jacobiSolver.validation(), true);
  ASSERT_ANY_THROW(jacobiSolver.pre_processing());
}

TEST(kozlova_e_jacobi_method, Test_1x1_System) {
  const int N = 1;

  std::vector<double> A = {1};
  std::vector<double> B = {5};
  std::vector<double> X(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->inputs_count.emplace_back(N);
  taskData->inputs_count.emplace_back(B.size());
  taskData->inputs_count.emplace_back(X.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->outputs_count.emplace_back(X.size());

  kozlova_e_jacobi_method::MethodJacobi jacobiSolver(taskData);
  ASSERT_EQ(jacobiSolver.validation(), true);
  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(X[0], 5.0, 1e-6);
}

TEST(kozlova_e_jacobi_method, Test_Zero_Right_Side) {
  const int N = 3;

  std::vector<double> A = {4, 1, 1, 1, 3, 1, 1, 1, 3};
  std::vector<double> B = {0, 0, 0};
  std::vector<double> X(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->inputs_count.emplace_back(N);
  taskData->inputs_count.emplace_back(B.size());
  taskData->inputs_count.emplace_back(X.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->outputs_count.emplace_back(X.size());

  kozlova_e_jacobi_method::MethodJacobi jacobiSolver(taskData);
  ASSERT_EQ(jacobiSolver.validation(), true);
  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_EQ(X[0], 0.0);
  ASSERT_EQ(X[1], 0.0);
  ASSERT_EQ(X[2], 0.0);
}

TEST(kozlova_e_jacobi_method, Test_Diagonal_Dominance_Large_Error) {
  const int N = 3;

  std::vector<double> A = {1000, 1, 1, 1, 1000, 1, 1, 1, 1000};
  std::vector<double> B = {3, 2.5, 2.5};
  std::vector<double> X(N, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->inputs_count.emplace_back(N);
  taskData->inputs_count.emplace_back(B.size());
  taskData->inputs_count.emplace_back(X.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskData->outputs_count.emplace_back(X.size());

  kozlova_e_jacobi_method::MethodJacobi jacobiSolver(taskData);
  ASSERT_EQ(jacobiSolver.validation(), true);
  jacobiSolver.pre_processing();
  jacobiSolver.run();
  jacobiSolver.post_processing();

  ASSERT_NEAR(X[0], 0.003, 1e-4);
  ASSERT_NEAR(X[1], 0.0025, 1e-4);
  ASSERT_NEAR(X[2], 0.0025, 1e-4);
}

