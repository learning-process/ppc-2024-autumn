// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/makhov_m_num_of_diff_elements_in_two_str/include/ops_seq.hpp"

TEST(makhov_m_num_of_diff_elements_in_two_str_seq, SameEmptySymbolsStrings) {
  std::string str1, str2;

  // Create data
  str1 = "   ";
  str2 = str1;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
  taskDataSeq->inputs_count.emplace_back(str2.size());
  taskDataSeq->inputs_count.emplace_back(str1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  int ref = 0;
  makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential testTaskmakhov_m_num_of_diff_elements_in_two_str_seq(
      taskDataSeq);
  ASSERT_EQ(testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.validation(), true);
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.pre_processing();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.run();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(makhov_m_num_of_diff_elements_in_two_str_seq, DiffSizeEmptySymbolsStrings) {
  std::string str1, str2;

  // Create data
  str1 = "   ";
  str2 = "      ";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
  taskDataSeq->inputs_count.emplace_back(str2.size());
  taskDataSeq->inputs_count.emplace_back(str1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  int ref = 3;
  makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential testTaskmakhov_m_num_of_diff_elements_in_two_str_seq(
      taskDataSeq);
  ASSERT_EQ(testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.validation(), true);
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.pre_processing();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.run();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(makhov_m_num_of_diff_elements_in_two_str_seq, EqualSizeDiffStrings) {
  std::string str1, str2;

  // Create data
  str1 = "Hello, World!!";
  str2 = "Goodbye, World";
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
  taskDataSeq->inputs_count.emplace_back(str2.size());
  taskDataSeq->inputs_count.emplace_back(str1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  int ref = str1.size();
  makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential testTaskmakhov_m_num_of_diff_elements_in_two_str_seq(
      taskDataSeq);
  ASSERT_EQ(testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.validation(), true);
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.pre_processing();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.run();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(makhov_m_num_of_diff_elements_in_two_str_seq, DiffSizeDiffStrings) {
  std::string str1, str2;

  // Create data
  str2 = "12341278";
  str1 = "12341278dfgdfg";
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
  taskDataSeq->inputs_count.emplace_back(str1.size());
  taskDataSeq->inputs_count.emplace_back(str2.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  int ref = 6;
  makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential testTaskmakhov_m_num_of_diff_elements_in_two_str_seq(
      taskDataSeq);
  ASSERT_EQ(testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.validation(), true);
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.pre_processing();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.run();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.post_processing();
  ASSERT_EQ(ref, out[0]);
}

TEST(makhov_m_num_of_diff_elements_in_two_str_seq, SameStrings) {
  std::string str1, str2;

  // Create data
  str1 = "Hello, World!";
  str2 = str1;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
  taskDataSeq->inputs_count.emplace_back(str2.size());
  taskDataSeq->inputs_count.emplace_back(str1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  int ref = 0;
  makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential testTaskmakhov_m_num_of_diff_elements_in_two_str_seq(
      taskDataSeq);
  ASSERT_EQ(testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.validation(), true);
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.pre_processing();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.run();
  testTaskmakhov_m_num_of_diff_elements_in_two_str_seq.post_processing();
  ASSERT_EQ(ref, out[0]);
}
