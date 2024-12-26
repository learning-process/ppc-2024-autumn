// Copyright 2024 Sdobnov Vladimir

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "seq/Sdobnov_V_iteration_method_yakoby/include/ops_seq.hpp"

TEST(Sdobnov_V_iteration_method_yakoby_seq, InvalidMatrixWithoutDiagonalDominance) {
  int size = 3;
  std::vector<double> matrix = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(free_members.data()));
  taskDataPar->outputs_count.emplace_back(size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq test(taskDataPar);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_iteration_method_yakoby_seq, InvalidInputCount) {
  int size = 3;
  std::vector<double> matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(free_members.data()));
  taskDataPar->outputs_count.emplace_back(size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq test(taskDataPar);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_iteration_method_yakoby_seq, InvalidInput) {
  int size = 3;
  std::vector<double> matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(-size);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq test(taskDataPar);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_iteration_method_yakoby_seq, InvalidOutputCount) {
  int size = 3;
  std::vector<double> matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(-size);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(free_members.data()));
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq test(taskDataPar);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_iteration_method_yakoby_seq, InvalidOutput) {
  int size = 3;
  std::vector<double> matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> free_members = {5.0, 5.0, 5.0};
  std::vector<double> res(size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(-size);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataPar->outputs_count.emplace_back(size);

  Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq test(taskDataPar);

  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_iteration_method_yakoby_seq, IterationMethodTest3x3) {
  int size = 3;
  std::vector<double> res(size, 0.0);
  std::vector<double> input_matrix = {2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0};
  std::vector<double> input_free_members = {6.0, 6.0, 6.0};
  std::vector<double> expected_res = {2.0, 2.0, 2.0};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
  taskDataPar->outputs_count.emplace_back(size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (int i = 0; i < size; i++) {
    ASSERT_NEAR(res[i], expected_res[i], 1e-3);
  }
}

TEST(Sdobnov_V_iteration_method_yakoby_seq, IterationMethodTest4x4) {
  int size = 4;
  std::vector<double> res(size, 0.0);
  std::vector<double> input_matrix = {12.0, 4.0, -2.0, 0.0, 2.0, 9.0,  3.0,  -3.0,
                                      -2.0, 1.0, 6.0,  2.0, 0.0, -1.0, -7.0, 10.0};
  std::vector<double> input_free_members = {10.0, 9.0, 8.0, 7.0};
  std::vector<double> expected_res = {0.637086, 1.036719, 0.895960, 1.430844};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(size);
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_free_members.data()));
  taskDataPar->outputs_count.emplace_back(size);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));

  Sdobnov_iteration_method_yakoby_seq::IterationMethodYakobySeq test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  for (int i = 0; i < size; i++) {
    ASSERT_NEAR(res[i], expected_res[i], 1e-3);
  }
}
