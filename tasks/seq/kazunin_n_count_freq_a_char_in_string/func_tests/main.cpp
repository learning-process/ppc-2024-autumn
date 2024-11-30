// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "seq/kazunin_n_count_freq_a_char_in_string/include/ops_seq.hpp"

TEST(kazunin_n_count_freq_a_char_in_string_seq, test_numeric_characters) {
  std::string test_string = "1122334455";

  char target_character = '2';
  int expected_count = 2;

  std::vector<std::string> input_strings(1, test_string);
  std::vector<char> target_characters(1, target_character);
  std::vector<int> output(1, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_strings.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_characters.data()));
  taskDataSeq->inputs_count.emplace_back(input_strings.size());
  taskDataSeq->inputs_count.emplace_back(target_characters.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential freq_char_task(taskDataSeq);
  ASSERT_TRUE(freq_char_task.validation());
  freq_char_task.pre_processing();
  freq_char_task.run();
  freq_char_task.post_processing();
  ASSERT_EQ(expected_count, output[0]);
}

TEST(kazunin_n_count_freq_a_char_in_string_seq, test_empty_string) {
  std::string test_string;

  char target_character = 'p';
  int expected_count = 0;

  std::vector<std::string> input_strings(1, test_string);
  std::vector<char> target_characters(1, target_character);
  std::vector<int> output(1, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_strings.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_characters.data()));
  taskDataSeq->inputs_count.emplace_back(input_strings.size());
  taskDataSeq->inputs_count.emplace_back(target_characters.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential freq_char_task(taskDataSeq);
  ASSERT_TRUE(freq_char_task.validation());
  freq_char_task.pre_processing();
  freq_char_task.run();
  freq_char_task.post_processing();
  ASSERT_EQ(expected_count, output[0]);
}

TEST(kazunin_n_count_freq_a_char_in_string_seq, test_mixed_characters) {
  std::string test_string = "a1b2c3d4a5";

  char target_character = 'a';
  int expected_count = 2;

  std::vector<std::string> input_strings(1, test_string);
  std::vector<char> target_characters(1, target_character);
  std::vector<int> output(1, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_strings.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_characters.data()));
  taskDataSeq->inputs_count.emplace_back(input_strings.size());
  taskDataSeq->inputs_count.emplace_back(target_characters.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential freq_char_task(taskDataSeq);
  ASSERT_TRUE(freq_char_task.validation());
  freq_char_task.pre_processing();
  freq_char_task.run();
  freq_char_task.post_processing();
  ASSERT_EQ(expected_count, output[0]);
}

TEST(kazunin_n_count_freq_a_char_in_string_seq, test_absent_character_in_repeated_string) {
  std::string test_string(500, 'x');

  char target_character = 'y';
  int expected_count = 0;

  std::vector<std::string> input_strings(1, test_string);
  std::vector<char> target_characters(1, target_character);
  std::vector<int> output(1, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_strings.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_characters.data()));
  taskDataSeq->inputs_count.emplace_back(input_strings.size());
  taskDataSeq->inputs_count.emplace_back(target_characters.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential freq_char_task(taskDataSeq);
  ASSERT_TRUE(freq_char_task.validation());
  freq_char_task.pre_processing();
  freq_char_task.run();
  freq_char_task.post_processing();
  ASSERT_EQ(expected_count, output[0]);
}

TEST(kazunin_n_count_freq_a_char_in_string_seq, test_special_characters) {
  std::string test_string = "@@##!!&&";

  char target_character = '#';
  int expected_count = 2;

  std::vector<std::string> input_strings(1, test_string);
  std::vector<char> target_characters(1, target_character);
  std::vector<int> output(1, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_strings.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_characters.data()));
  taskDataSeq->inputs_count.emplace_back(input_strings.size());
  taskDataSeq->inputs_count.emplace_back(target_characters.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  kazunin_n_count_freq_a_char_in_string_seq::CountFreqCharTaskSequential freq_char_task(taskDataSeq);
  ASSERT_TRUE(freq_char_task.validation());
  freq_char_task.pre_processing();
  freq_char_task.run();
  freq_char_task.post_processing();

  ASSERT_EQ(expected_count, output[0]);
}
