#include <gtest/gtest.h>

#include <vector>

#include "seq/chernykh_a_num_of_alternations_signs/include/ops_seq.hpp"

TEST(chernykh_a_num_of_alternations_signs_seq, correct_alternating_signs_count) {
  // Create data
  auto input = std::vector<int>{3, -2, 4, -5, -1, 6};
  auto output = std::vector<int>(1, 0);
  auto want = 4;

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  // Create Task
  auto task = chernykh_a_num_of_alternations_signs_seq::Task(task_data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  ASSERT_EQ(want, output[0]);
}

TEST(chernykh_a_num_of_alternations_signs_seq, input_size_less_than_two_fails_validation) {
  // Create data
  auto input = std::vector<int>();
  auto output = std::vector<int>(1, 0);

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  // Create Task
  auto task = chernykh_a_num_of_alternations_signs_seq::Task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(chernykh_a_num_of_alternations_signs_seq, output_size_not_equal_one_fails_validation) {
  // Create data
  auto input = std::vector<int>{3, -2, 4, -5, -1, 6};
  auto output = std::vector<int>();

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  // Create Task
  auto task = chernykh_a_num_of_alternations_signs_seq::Task(task_data);

  ASSERT_FALSE(task.validation());
}

TEST(chernykh_a_num_of_alternations_signs_seq, all_elements_are_equal) {
  // Create data
  auto input = std::vector<int>(5, 0);
  auto output = std::vector<int>(1, 0);
  auto want = 0;

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  // Create Task
  auto task = chernykh_a_num_of_alternations_signs_seq::Task(task_data);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  ASSERT_EQ(want, output[0]);
}
