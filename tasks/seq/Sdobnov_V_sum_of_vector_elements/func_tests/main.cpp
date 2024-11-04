// Copyright 2024 Sdobnov Vladimir

#include <gtest/gtest.h>

#include <vector>

#include "seq/Sdobnov_V_sum_of_vector_elements/include/ops_seq.hpp"

TEST(Sdobnov_V_sum_of_vector_elements_seq, EmptyInput) {
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);
  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, EmptyOutput) {
  int rows = 10;
  int columns = 10;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);
  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, EmptyMatrix) {
  int rows = 0;
  int columns = 0;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(0, res);
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, Matrix10x10) {
  int rows = 10;
  int columns = 10;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int>& vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(sum, res);
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, Matrix100x100) {
  int rows = 100;
  int columns = 100;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int>& vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(sum, res);
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, Matrix100x10) {
  int rows = 100;
  int columns = 10;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int>& vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(sum, res);
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, Matrix10x100) {
  int rows = 10;
  int columns = 100;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int>& vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(sum, res);
}