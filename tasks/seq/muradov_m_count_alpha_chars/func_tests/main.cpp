#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "seq/muradov_m_count_alpha_chars/include/ops_seq.hpp"

int count_alpha_chars(const std::string& str) {
  return std::count_if(str.begin(), str.end(), [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });
}

TEST(muradov_m_count_alpha_chars_seq, test_empty_string) {
  std::string input_str;
  int expected_alpha_count = 0;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected_alpha_count, out[0]);
}

TEST(muradov_m_count_alpha_chars_seq, test_only_non_alpha_characters) {
  std::string input_str = "1234567890!@#$%^&*()";
  int expected_alpha_count = 0;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected_alpha_count, out[0]);
}

TEST(muradov_m_count_alpha_chars_seq, test_mixed_alpha_and_non_alpha) {
  std::string input_str = "Hello, World! 123";
  int expected_alpha_count = count_alpha_chars(input_str);

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected_alpha_count, out[0]);
}

TEST(muradov_m_count_alpha_chars_seq, test_all_alpha_characters) {
  std::string input_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  int expected_alpha_count = count_alpha_chars(input_str);

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected_alpha_count, out[0]);
}

TEST(muradov_m_count_alpha_chars_seq, test_large_input_string) {
  std::string input_str = std::string(100000, 'a') + std::string(100000, '1');
  int expected_alpha_count = count_alpha_chars(input_str);

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected_alpha_count, out[0]);
}
