// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "seq/naumov_b_bubble_sort/include/ops_seq.hpp"

TEST(naumov_b_bubble_sort_seq, TestValidationFailure) {
  std::vector<int> in;
  std::vector<int> out(10);

  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  tmpPar->inputs_count.emplace_back(in.size());
  tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));  // Использование reinterpret_cast
  tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  tmpPar->outputs_count.emplace_back(out.size());

  naumov_b_bubble_sort_seq::TestTaskSequential tmpTaskPar(tmpPar);

  ASSERT_FALSE(tmpTaskPar.validation());
}

TEST(naumov_b_bubble_sort_seq, sorted_input) {
  std::vector<int> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> output_data(input_data.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  naumov_b_bubble_sort_seq::TestTaskSequential sort_task(task_data);

  ASSERT_TRUE(sort_task.validation());
  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  ASSERT_EQ(output_data, input_data);
}

TEST(naumov_b_bubble_sort_seq, random_input_10) {
  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int> distribution(-1000, 1000);

  std::vector<int> input_data(10);
  for (size_t i = 0; i < 10; ++i) {
    input_data[i] = distribution(generator);
  }

  std::vector<int> output_data(input_data.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  naumov_b_bubble_sort_seq::TestTaskSequential sort_task(task_data);

  ASSERT_TRUE(sort_task.validation());
  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  std::vector<int> reference_data = input_data;
  std::sort(reference_data.begin(), reference_data.end());
  ASSERT_EQ(output_data, reference_data);
}

TEST(naumov_b_bubble_sort_seq, Test_RepeatedElements) {
  std::vector<int> input_data = {7, 7, 7, 7, 7};
  std::vector<int> output_data(input_data.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  naumov_b_bubble_sort_seq::TestTaskSequential sort_task(task_data);

  ASSERT_TRUE(sort_task.validation());
  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  ASSERT_EQ(output_data, input_data);
}

TEST(naumov_b_bubble_sort_seq, Test_ReverseOrder) {
  std::vector<int> input_data = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> sorted_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int> output_data(input_data.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  naumov_b_bubble_sort_seq::TestTaskSequential sort_task(task_data);

  ASSERT_TRUE(sort_task.validation());
  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  ASSERT_EQ(output_data, sorted_data);
}
