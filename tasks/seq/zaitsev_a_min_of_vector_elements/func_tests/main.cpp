// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/zaitsev_a_min_of_vector_elements/include/ops_seq.hpp"

using namespace std::chrono_literals;

TEST(zaitsev_a_min_of_vector_elements_sequentional, test_length_10) {
  const int length = 10;
  const int extrema = -1;
  const int minRangeValue = 100;
  const int maxRangeValue = 1000;

  std::mt19937 gen(31415);

  // Create data
  std::vector<int> in(length);
  for (size_t i = 0; i < length; i++) {
    int j = minRangeValue + gen() % (maxRangeValue - minRangeValue + 1);
    in[i] = j;
  }
  in[length / 2] = extrema;

  std::vector<int> out(1, extrema);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(extrema, out[0]);
}
TEST(zaitsev_a_min_of_vector_elements_sequentional, test_length_50) {
  const int length = 50;
  const int extrema = -1;
  const int minRangeValue = 100;
  const int maxRangeValue = 1000;

  std::mt19937 gen(31415);

  // Create data
  std::vector<int> in(length);
  for (size_t i = 0; i < length; i++) {
    int j = minRangeValue + gen() % (maxRangeValue - minRangeValue + 1);
    in[i] = j;
  }
  in[length / 2] = extrema;
  std::vector<int> out(1, extrema);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(extrema, out[0]);
}

TEST(zaitsev_a_min_of_vector_elements_sequentional, test_length_1) {
  const int length = 1;
  const int extrema = -1;

  std::mt19937 gen(31415);

  // Create data
  std::vector<int> in(length, extrema);
  std::vector<int> out(1, extrema);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(extrema, out[0]);
}

TEST(zaitsev_a_min_of_vector_elements_sequentional, test_vector_of_negative_elements) {
  const int length = 10;
  const int extrema = -105;
  const int minRangeValue = -100;
  const int maxRangeValue = -1;

  std::mt19937 gen(31415);

  // Create data
  std::vector<int> in(length);
  for (size_t i = 0; i < length; i++) {
    int j = minRangeValue + gen() % (maxRangeValue - minRangeValue + 1);
    in[i] = j;
  }
  in[length / 2] = extrema;
  std::vector<int> out(1, extrema);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  zaitsev_a_min_of_vector_elements_seq::MinOfVectorElementsSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(extrema, out[0]);
}