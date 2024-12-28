// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/matyunina_a_batcher_qsort/include/ops_seq.hpp"

namespace matyunina_a_batcher_qsort_seq {

std::vector<int> generateRandomVector(size_t size, int min_value, int max_value) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(min_value, max_value);

  std::vector<int> in(size);
  for (size_t i = 0; i < size; i++) {
    in[i] = dist(gen);
  }

  return in;
}

void run_test(std::vector<int32_t>& in) {
  std::vector<int32_t> out(in.size());
  std::vector<int32_t> sorted(in);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskData->inputs_count.emplace_back(in.size());

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  matyunina_a_batcher_qsort_seq::TestTaskSequential<int32_t> testTaskSequential(taskData);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  std::sort(sorted.begin(), sorted.end());
  ASSERT_EQ(sorted, out);
}
}  // namespace matyunina_a_batcher_qsort_seq

TEST(matyunina_a_batcher_qsort_seq, random_vector) {
  int size = 10;
  int min = -500;
  int max = 500;
  std::vector<int32_t> in = matyunina_a_batcher_qsort_seq::generateRandomVector(size, min, max);
  matyunina_a_batcher_qsort_seq::run_test(in);
}

// Тест пустого вектора
TEST(matyunina_a_batcher_qsort_seq, zero) {
  std::vector<int> in;
  matyunina_a_batcher_qsort_seq::run_test(in);
}

// Тест вектора с одним элементом
TEST(matyunina_a_batcher_qsort_seq, single) {
  std::vector<int> in = {42};
  matyunina_a_batcher_qsort_seq::run_test(in);
}

// Тест вектора с повторяющимися элементами
TEST(matyunina_a_batcher_qsort_seq, duplicated_elements) {
  std::vector<int32_t> in = {3, 4, 4, 1, 5, 5, 2, 6, 5, 3};
  matyunina_a_batcher_qsort_seq::run_test(in);
}

TEST(matyunina_a_batcher_qsort_seq, video_example) {
  std::vector<int> in = {8, 2, 5, 10, 1, 7, 3, 12, 6, 11, 4, 9};
  matyunina_a_batcher_qsort_seq::run_test(in);
}