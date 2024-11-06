

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "seq/strakhov_a_str_char_freq/include/ops_seq.hpp"

TEST(strakhov_a_str_char_freq_seq, simple_test) {
  std::string input = "0123456789";
  char target = '7';
  float expected_res = 0.1f;
  std::vector<std::string> input_vec(1, input);
  std::vector<char> target_vec(1, target);
  std::vector<int> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->inputs_count.emplace_back(target_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());
  strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_res, res[0]);
}

TEST(strakhov_a_str_char_freq_seq, test_full) {
  std::string input = "111111";
  char target = '1';
  float expected_res = 1;
  std::vector<std::string> input_vec(1, input);
  std::vector<char> target_vec(1, target);
  std::vector<int> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->inputs_count.emplace_back(target_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());
  strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_res, res[0]);
}

TEST(strakhov_a_str_char_freq_seq, test_half) {
  std::string input = "111222";
  char target = '1';
  float expected_res = 0.5f;
  std::vector<std::string> input_vec(1, input);
  std::vector<char> target_vec(1, target);
  std::vector<int> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->inputs_count.emplace_back(target_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());
  strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_res, res[0]);
}

TEST(strakhov_a_str_char_freq_seq, test_empty) {
  std::string input = "222222";
  char target = '1';
  float expected_res = 0;
  std::vector<std::string> input_vec(1, input);
  std::vector<char> target_vec(1, target);
  std::vector<int> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->inputs_count.emplace_back(target_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());
  strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_res, res[0]);
}

TEST(strakhov_a_str_char_freq_seq, test_null) {
  std::string input;
  char target = '1';
  float expected_res = 0;
  std::vector<std::string> input_vec(1, input);
  std::vector<char> target_vec(1, target);
  std::vector<int> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->inputs_count.emplace_back(target_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());
  strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential charFrequencyTask(taskDataSeq);
  ASSERT_EQ(charFrequencyTask.validation(), true);
  charFrequencyTask.pre_processing();
  charFrequencyTask.run();
  charFrequencyTask.post_processing();
  ASSERT_EQ(expected_res, res[0]);
}
