// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/rams_s_char_frequency/include/ops_seq.hpp"

TEST(rams_s_char_frequency_seq, several_occurrences_of_target) {
  std::string in = "abcdabcda";
  std::vector<int> in_target(1, 'a');
  std::vector<int> out(1, 0);
  int expected_count = 3;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  taskDataSeq->inputs_count.emplace_back(in_target.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rams_s_char_frequency_seq::CharFrequencyTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(rams_s_char_frequency_seq, no_occurrences_of_target) {
  std::string in = "bcdbcd";
  std::vector<int> in_target(1, 'a');
  std::vector<int> out(1, 0);
  int expected_count = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  taskDataSeq->inputs_count.emplace_back(in_target.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rams_s_char_frequency_seq::CharFrequencyTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(rams_s_char_frequency_seq, empty_input_string) {
  std::string in;
  std::vector<int> in_target(1, 'a');
  std::vector<int> out(1, 0);
  int expected_count = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  taskDataSeq->inputs_count.emplace_back(in_target.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rams_s_char_frequency_seq::CharFrequencyTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}

TEST(rams_s_char_frequency_seq, large_input_string) {
  std::string common_string = "abc";
  std::string in;
  for (int i = 0; i < 9999; i++) {
    in += common_string;
  }
  std::vector<int> in_target(1, 'a');
  std::vector<int> out(1, 0);
  int expected_count = 9999;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_target.data()));
  taskDataSeq->inputs_count.emplace_back(in_target.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rams_s_char_frequency_seq::CharFrequencyTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expected_count, out[0]);
}
