// Copyright 2024 Anikin Maksim
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "seq/anikin_m_summ_of_different_symbols/include/ops_seq.hpp"

TEST(SumDifSymSequential_count, ans_0) {
  // Create Data
  std::string str1 = "abcd";
  std::string str2 = "abcd";

  std::vector<std::string> to_input_vector{str1, str2};
  std::vector<int> out(1,0);

  // Create Task Data
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(to_input_vector[0]);
  taskDataSeq->inputs.emplace_back(to_input_vector[1]);
  taskDataSeq->inputs_count.emplace_back(to_input_vector.size());
  taskDataSeq->outputs.emplace_back(out[0]);
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  anikin_m_sum_of_differnt_symbols_seq::SumDifSymSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(0, out[0]);
}
/*
TEST(SumDifSymSequential_count, ans_1) {
  // Create Data
  std::string str1 = "abcde";
  std::string str2 = "abcd";

  std::vector<std::string> to_input_vector{str1, str2};
  std::vector<int> out(1,1);

  // Create Task Data
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<char*>(to_input_vector[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<char*>(to_input_vector[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  anikin_m_sum_of_differnt_symbols_seq::SumDifSymSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(1, out[0]);
}

TEST(SumDifSymSequential_count, ans_2) {
  // Create Data
  std::string str1 = "abcd";
  std::string str2 = "abcded";

  std::vector<std::string> to_input_vector{str1, str2};
  std::vector<int> out(1,1);

  // Create Task Data
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<char*>(to_input_vector[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<char*>(to_input_vector[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  anikin_m_sum_of_differnt_symbols_seq::SumDifSymSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(2, out[0]);
}

TEST(SumDifSymSequential_count, ans_6) {
  // Create Data
  std::string str1 = "zxvn";
  std::string str2 = "abcded";

  std::vector<std::string> to_input_vector{str1, str2};
  std::vector<int> out(1,1);

  // Create Task Data
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<char*>(to_input_vector[0]));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<char*>(to_input_vector[1]));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  anikin_m_sum_of_differnt_symbols_seq::SumDifSymSequential testClass(taskDataSeq);
  ASSERT_EQ(testClass.validation(), true);
  testClass.pre_processing();
  testClass.run();
  testClass.post_processing();
  ASSERT_EQ(6, out[0]);
}
*/
