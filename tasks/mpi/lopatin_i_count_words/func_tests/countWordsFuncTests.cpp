#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <string>

#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"


TEST(lopatin_i_count_words_mpi, test_empty_string) {
  boost::mpi::communicator world;
  std::string input = "";
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.c_str())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
    taskData->outputs_count.emplace_back(word_count.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), false);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(word_count[0], 0);
  }
}

TEST(lopatin_i_count_words_mpi, test_single_word) {
  boost::mpi::communicator world;
  std::string input = "Hello";
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.c_str())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
    taskData->outputs_count.emplace_back(word_count.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(word_count[0], 1);
  }
}

TEST(lopatin_i_count_words_mpi, test_multiple_words) {
  boost::mpi::communicator world;
  std::string input = "This is a test sentence";
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.c_str())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
    taskData->outputs_count.emplace_back(word_count.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(word_count[0], 5);
  }
}

TEST(lopatin_i_count_words_mpi, test_multiple_sentences) {
  boost::mpi::communicator world;
  std::string input = "This is a test sentence. This is another one. And one more. And another one.";
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input.c_str())));
    taskData->inputs_count.emplace_back(input.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
    taskData->outputs_count.emplace_back(word_count.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTask(taskData);
  ASSERT_EQ(testTask.validation(), true);
  testTask.pre_processing();
  testTask.run();
  testTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(word_count[0], 15);
  }
}
