#include <gtest/gtest.h>

#include "seq/kapustin_i_bubble/include/avg_seq.hpp"

static std::vector<int> generate_random_data(int size, int min_val, int max_val) {
  std::vector<int> data(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min_val, max_val);

  for (int i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }

  return data;
}
TEST(kapustin_i_bubble_sort_seq, Random_test_50_negat_val) {
  int total_elements = 50;
  std::vector<int> input_data = generate_random_data(total_elements, -100, 100);

  std::vector<int> expected_output = input_data;
  std::sort(expected_output.begin(), expected_output.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(total_elements);
  std::vector<int> output_data(total_elements);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(total_elements);

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);

  ASSERT_TRUE(bubbleSort.validation());
  bubbleSort.pre_processing();
  bubbleSort.run();
  bubbleSort.post_processing();

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], total_elements * sizeof(int));

  ASSERT_EQ(output_data, expected_output);
}
TEST(kapustin_i_bubble_sort_seq, Random_test_50_small_val) {
  int total_elements = 50;
  std::vector<int> input_data = generate_random_data(total_elements, 1, 1000);

  std::vector<int> expected_output = input_data;
  std::sort(expected_output.begin(), expected_output.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(total_elements);
  std::vector<int> output_data(total_elements);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(total_elements);

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);

  ASSERT_TRUE(bubbleSort.validation());
  bubbleSort.pre_processing();
  bubbleSort.run();
  bubbleSort.post_processing();

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], total_elements * sizeof(int));

  ASSERT_EQ(output_data, expected_output);
}
TEST(kapustin_i_bubble_sort_seq, Random_test_5000_large_val) {
  int total_elements = 5000;
  std::vector<int> input_data = generate_random_data(total_elements, 10000, 10000000);

  std::vector<int> expected_output = input_data;
  std::sort(expected_output.begin(), expected_output.end());

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(total_elements);
  std::vector<int> output_data(total_elements);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(total_elements);

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);

  ASSERT_TRUE(bubbleSort.validation());
  bubbleSort.pre_processing();
  bubbleSort.run();
  bubbleSort.post_processing();

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], total_elements * sizeof(int));

  ASSERT_EQ(output_data, expected_output);
}

TEST(kapustin_i_bubble_sort_seq, simple_test_1) {
  std::vector<int> input_data = {5, 2, 9, 1, 5, 6};

  std::vector<int> expected_output = {1, 2, 5, 5, 6, 9};

  int total_elements = input_data.size();

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(total_elements);
  std::vector<int> output_data(total_elements);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(total_elements);

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);

  ASSERT_TRUE(bubbleSort.validation());

  bubbleSort.pre_processing();
  bubbleSort.run();
  bubbleSort.post_processing();

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], total_elements * sizeof(int));

  ASSERT_EQ(output_data, expected_output);
}

TEST(kapustin_i_bubble_sort_seq, simple_test_2) {
  std::vector<int> input_data = {12, 7, 9, 18, 2, 5, 14, 8, 1, 6};
  std::vector<int> expected_output = {1, 2, 5, 6, 7, 8, 9, 12, 14, 18};
  int total_elements = input_data.size();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(total_elements);
  std::vector<int> output_data(total_elements);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(total_elements);

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);
  ASSERT_TRUE(bubbleSort.validation());
  bubbleSort.pre_processing();
  bubbleSort.run();
  bubbleSort.post_processing();

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], total_elements * sizeof(int));
  ASSERT_EQ(output_data, expected_output);
}

TEST(kapustin_i_bubble_sort_seq, Validation_EmptyInputs) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);

  ASSERT_FALSE(bubbleSort.validation());
}
TEST(kapustin_i_bubble_sort_seq, sorted_before) {
  std::vector<int> input_data = {1, 3, 5, 6, 7, 8, 100, 1000};
  std::vector<int> expected_output = {1, 3, 5, 6, 7, 8, 100, 1000};
  int total_elements = input_data.size();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(total_elements);
  std::vector<int> output_data(total_elements);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(total_elements);

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);
  ASSERT_TRUE(bubbleSort.validation());
  bubbleSort.pre_processing();
  bubbleSort.run();
  bubbleSort.post_processing();

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], total_elements * sizeof(int));
  ASSERT_EQ(output_data, expected_output);
}

TEST(kapustin_i_bubble_sort_seq, eq_val) {
  std::vector<int> input_data = {5, 5, 5, 5, 5, 5, 5, 5, 5};
  std::vector<int> expected_output = {5, 5, 5, 5, 5, 5, 5, 5, 5};
  int total_elements = input_data.size();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(total_elements);
  std::vector<int> output_data(total_elements);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(total_elements);

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);
  ASSERT_TRUE(bubbleSort.validation());
  bubbleSort.pre_processing();
  bubbleSort.run();
  bubbleSort.post_processing();

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], total_elements * sizeof(int));
  ASSERT_EQ(output_data, expected_output);
}

TEST(kapustin_i_bubble_sort_seq, some_eq_val) {
  std::vector<int> input_data = {1, 5, 5, 5, 3, 10, 2, 1, 5};
  std::vector<int> expected_output = {1, 1, 2, 3, 5, 5, 5, 5, 10};
  int total_elements = input_data.size();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(total_elements);
  std::vector<int> output_data(total_elements);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskDataSeq->outputs_count.emplace_back(total_elements);

  kapustin_i_bubble_sort_seq::BubbleSortSequential bubbleSort(taskDataSeq);
  ASSERT_TRUE(bubbleSort.validation());
  bubbleSort.pre_processing();
  bubbleSort.run();
  bubbleSort.post_processing();

  std::memcpy(output_data.data(), taskDataSeq->outputs[0], total_elements * sizeof(int));
  ASSERT_EQ(output_data, expected_output);
}