#include <gtest/gtest.h>

#include "seq/chastov_v_bubble_sort/include/ops_seq.hpp"

TEST(chastov_v_bubble_sort, zero_len) {
  std::vector<int> input_data;
  std::vector<int> output_data(0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->inputs.emplace_back(input_data.empty() ? nullptr : reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->outputs.emplace_back(output_data.empty() ? nullptr : reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(output_data.size());

  chastov_v_bubble_sort::TestTaskSequential<int> tmpTaskSeq(taskDataSeq);

  ASSERT_FALSE(tmpTaskSeq.validation());
}

TEST(chastov_v_bubble_sort, sorted_input) {
  std::vector<int> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> output_data(input_data.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  chastov_v_bubble_sort::TestTaskSequential<int> sort_task(task_data);

  ASSERT_TRUE(sort_task.validation());

  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  ASSERT_EQ(output_data, input_data);
}

TEST(chastov_v_bubble_sort, reverse_sorted_input) {
  std::vector<int> input_data = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> output_data(input_data.size(), 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  chastov_v_bubble_sort::TestTaskSequential<int> sort_task(task_data);

  ASSERT_TRUE(sort_task.validation());

  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  std::vector<int> expected_data = input_data;
  std::sort(expected_data.begin(), expected_data.end());

  ASSERT_EQ(output_data, expected_data);
}

TEST(chastov_v_bubble_sort, test_int_rand_120) {
  constexpr size_t mass_size = 120;
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  std::vector<int> input_data(mass_size);
  std::generate(input_data.begin(), input_data.end(), []() { return std::rand() * (std::rand() % 2 == 0 ? 1 : -1); });

  std::vector<int> output_data(mass_size);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.push_back(input_data.size());
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.push_back(output_data.size());

  chastov_v_bubble_sort::TestTaskSequential<int> sort_task(taskDataSeq);
  ASSERT_TRUE(sort_task.validation());

  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  std::vector<int> expected_data = input_data;
  std::sort(expected_data.begin(), expected_data.end());

  ASSERT_EQ(output_data, expected_data);
}

TEST(chastov_v_bubble_sort, test_int_rand_1200) {
  constexpr size_t mass_size = 1200;
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  std::vector<int> input_data(mass_size);
  std::generate(input_data.begin(), input_data.end(), []() { return std::rand() * (std::rand() % 2 == 0 ? 1 : -1); });

  std::vector<int> output_data(mass_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  chastov_v_bubble_sort::TestTaskSequential<int> sort_task(task_data);
  ASSERT_TRUE(sort_task.validation());

  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  std::vector<int> expected_data = input_data;
  std::sort(expected_data.begin(), expected_data.end());

  ASSERT_EQ(output_data, expected_data);
}

TEST(chastov_v_bubble_sort, test_double_rand_120) {
  constexpr size_t mass_size = 120;
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  std::vector<double> input_data(mass_size);
  std::generate(input_data.begin(), input_data.end(), []() {
    return static_cast<double>(std::rand()) / (std::rand() % 100 + 1) * (std::rand() % 2 == 0 ? 1.0 : -1.0);
  });

  std::vector<double> output_data(mass_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  chastov_v_bubble_sort::TestTaskSequential<double> sort_task(task_data);
  ASSERT_TRUE(sort_task.validation());

  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  std::vector<double> expected_data = input_data;
  std::sort(expected_data.begin(), expected_data.end());

  ASSERT_EQ(output_data, expected_data);
}

TEST(chastov_v_bubble_sort, test_double_rand_1200) {
  constexpr size_t mass_size = 1200;
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  std::vector<double> input_data(mass_size);
  std::generate(input_data.begin(), input_data.end(), []() {
    return static_cast<double>(std::rand()) / (std::rand() % 100 + 1) * (std::rand() % 2 == 0 ? 1.0 : -1.0);
  });

  std::vector<double> output_data(mass_size);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs_count.push_back(input_data.size());
  task_data->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  task_data->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  task_data->outputs_count.push_back(output_data.size());

  chastov_v_bubble_sort::TestTaskSequential<double> sort_task(task_data);
  ASSERT_TRUE(sort_task.validation());

  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  std::vector<double> expected_data = input_data;
  std::sort(expected_data.begin(), expected_data.end());

  ASSERT_EQ(output_data, expected_data);
}

TEST(chastov_v_bubble_sort, test_mass_identical_values) {
  constexpr size_t mass_size = 10;
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  const int constant_value = std::rand();

  std::vector<int> input_data(mass_size, constant_value);
  std::vector<int> output_data(mass_size);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.push_back(input_data.size());
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.push_back(output_data.size());

  chastov_v_bubble_sort::TestTaskSequential<int> sort_task(taskDataSeq);
  ASSERT_TRUE(sort_task.validation());

  sort_task.pre_processing();
  sort_task.run();
  sort_task.post_processing();

  int mismatched_elements =
      std::count_if(output_data.begin(), output_data.end(),
                    [&input_data, idx = 0](int value) mutable { return value != input_data[idx++]; });

  ASSERT_EQ(mismatched_elements, 0);
}