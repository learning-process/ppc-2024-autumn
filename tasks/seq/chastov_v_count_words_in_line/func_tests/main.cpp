#include <gtest/gtest.h>

#include "seq/chastov_v_count_words_in_line/include/ops_seq.hpp"
#include "seq/chastov_v_count_words_in_line/src/ops_seq.cpp"

// Test case to check the behavior of the word counting function when given an empty string
TEST(chastov_v_count_words_in_line_seq, empty_string) {
  std::vector<char> input = {};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  chastov_v_count_words_in_line_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), false);
}

// Test case to verify that the function correctly identifies that a string consisting only of spaces
TEST(chastov_v_count_words_in_line_seq, handles_only_spaces) {
  std::vector<char> inputData = {' ', ' ', ' '};
  std::vector<int> outputData(1, 0);

  auto taskDataPtr = std::make_shared<ppc::core::TaskData>();
  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskDataPtr->inputs_count.emplace_back(inputData.size());
  taskDataPtr->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskDataPtr->outputs_count.emplace_back(outputData.size());

  chastov_v_count_words_in_line_seq::TestTaskSequential wordCountTask(taskDataPtr);
  ASSERT_TRUE(wordCountTask.validation());
  wordCountTask.pre_processing();
  wordCountTask.run();
  wordCountTask.post_processing();

  ASSERT_EQ(outputData[0], 0);
}

// Test case to check the counting functionality for a single word input
TEST(chastov_v_count_words_in_line_seq, word_1) {
  std::vector<char> input;
  std::string testString = "hello";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    input.push_back(testString[j]);
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  chastov_v_count_words_in_line_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], 1);
}

// Test case for counting the number of words in a four word sentence
TEST(chastov_v_count_words_in_line_seq, words_4) {
  std::vector<char> input;
  std::string testString = "My name is Slava";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    input.push_back(testString[j]);
  }
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  chastov_v_count_words_in_line_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], 4);
}

// Test case to verify the function's ability to handle larger input sizes
// The generated string should contain enough words to yield a count of 450
TEST(chastov_v_count_words_in_line_seq, words_450) {
  std::vector<char> input = createTestInput(30);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  chastov_v_count_words_in_line_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], 450);
}

// Test case to check the performance and correctness for an even larger input size
// The created string should contain enough words to yield a count of 1500
TEST(chastov_v_count_words_in_line_seq, words_1500) {
  std::vector<char> input = createTestInput(100);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  chastov_v_count_words_in_line_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], 1500);
}

// Test case to evaluate the handling of a very large number of words
// The generated string should be such that the word count is expected to be 7500
TEST(chastov_v_count_words_in_line_seq, words_7500) {
  std::vector<char> input = createTestInput(500);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  chastov_v_count_words_in_line_seq::TestTaskSequential testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  ASSERT_EQ(out[0], 7500);
}

// Test case to check the counting of words that include special characters
// The input contains two words separated by a space, and the expected output is 2
TEST(chastov_v_count_words_in_line_seq, words_with_special_characters) {
  std::vector<char> inputData = {'W', 'o', 'r', 'd', '@', '1', ' ', 'W', 'o', 'r', 'd', '#', '2'};
  std::vector<int> outputData(1, 0);

  auto taskDataPtr = std::make_shared<ppc::core::TaskData>();
  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputData.data()));
  taskDataPtr->inputs_count.emplace_back(inputData.size());
  taskDataPtr->outputs.emplace_back(reinterpret_cast<uint8_t *>(outputData.data()));
  taskDataPtr->outputs_count.emplace_back(outputData.size());

  chastov_v_count_words_in_line_seq::TestTaskSequential wordCountTask(taskDataPtr);
  ASSERT_TRUE(wordCountTask.validation());
  wordCountTask.pre_processing();
  wordCountTask.run();
  wordCountTask.post_processing();

  ASSERT_EQ(outputData[0], 2);
}