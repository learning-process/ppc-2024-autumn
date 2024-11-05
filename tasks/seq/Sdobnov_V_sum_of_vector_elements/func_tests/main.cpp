// Copyright 2024 Sdobnov Vladimir

#include <gtest/gtest.h>

#include <vector>

#include "seq/Sdobnov_V_sum_of_vector_elements/include/ops_seq.hpp"

std::vector<int> generate_random_vector(int size, int lower_bound = 0, int upper_bound = 50) {
  std::vector<int> res(size);
  for (int i = 0; i < size; i++) {
    res[i] = lower_bound + rand() % (upper_bound - lower_bound + 1);
  }
  return res;
}

std::vector<std::vector<int>> generate_random_matrix(int rows, int columns, int lower_bound = 0, int upper_bound = 50) {
  std::vector<std::vector<int>> res(rows);
  for (int i = 0; i < rows; i++) {
    res[i] = generate_random_vector(columns, lower_bound, upper_bound);
  }
  return res;
  return std::vector<std::vector<int>>();
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, EmptyInput) {
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);
  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, EmptyOutput) {
  int rows = 10;
  int columns = 10;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);
  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, EmptyMatrix) {
  int rows = 0;
  int columns = 0;
  int res;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(0, res);
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, Matrix1x1) {
  int rows = 1;
  int columns = 1;
  int res;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(sum, res);
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, Matrix5x1) {
  int rows = 5;
  int columns = 1;
  int res;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(sum, res);
}

TEST(Sdobnov_V_sum_of_vector_elements_seq, Matrix10x10) {
  int rows = 10;
  int columns = 10;
  int res;
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

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
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

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
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

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
  std::vector<std::vector<int>> input = generate_random_matrix(rows, columns);
  int sum = 0;
  for (const std::vector<int> &vec : input) {
    for (int elem : vec) {
      sum += elem;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(rows);
  taskDataPar->inputs_count.emplace_back(columns);
  for (long unsigned int i = 0; i < input.size(); i++) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input[i].data()));
  }
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  Sdobnov_V_sum_of_vector_elements::SumVecElemSequential test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(sum, res);
}