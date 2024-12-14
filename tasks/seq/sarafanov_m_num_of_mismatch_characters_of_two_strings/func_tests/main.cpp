#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "seq/sarafanov_m_num_of_mismatch_characters_of_two_strings/include/ops_seq.hpp"

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, typical_scenario) {
  auto input_a = std::string("abcdefg");
  auto input_b = std::string("abcxyzg");
  auto output = std::vector<int>(1);
  auto expected = 3;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask(task_data);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected, output[0]);
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, with_special_characters_and_different_cases) {
  auto input_a = std::string("abc!@#123DEFghijklmn");
  auto input_b = std::string("abc$%^456XYZghijklMn");
  auto output = std::vector<int>(1);
  auto expected = 10;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask(task_data);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected, output[0]);
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, both_strings_are_empty) {
  auto input_a = std::string("");
  auto input_b = std::string("");
  auto output = std::vector<int>(1);
  auto expected = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask(task_data);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected, output[0]);
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, all_matching_characters) {
  auto input_a = std::string("aaaaaaaaaa");
  auto input_b = std::string("aaaaaaaaaa");
  auto output = std::vector<int>(1);
  auto expected = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask(task_data);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected, output[0]);
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, all_different_characters) {
  auto input_a = std::string("abcdefghij");
  auto input_b = std::string("klmnopqrst");
  auto output = std::vector<int>(1);
  auto expected = 10;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask(task_data);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  ASSERT_EQ(expected, output[0]);
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, error_when_input_lengths_are_different) {
  auto input_a = std::string("abcdefg");
  auto input_b = std::string("abc");
  auto output = std::vector<int>(1);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask(task_data);
  ASSERT_FALSE(task.validation());
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, error_when_output_size_is_not_equal_to_one) {
  auto input_a = std::string("abcdefg");
  auto input_b = std::string("abcxyzg");
  auto output = std::vector<int>(0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask(task_data);
  ASSERT_FALSE(task.validation());
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, error_when_one_string_is_empty) {
  auto input_a = std::string("abcdefg");
  auto input_b = std::string("");
  auto output = std::vector<int>(1);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask(task_data);
  ASSERT_FALSE(task.validation());
}
