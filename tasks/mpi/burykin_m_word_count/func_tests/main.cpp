#include <gtest/gtest.h>

#include "mpi/burykin_m_word_count/include/ops_mpi.hpp"

namespace burykin_m_word_count {

// Тестирование пустой строки
TEST(CountWordsMPI, TestEmptyString) {
  boost::mpi::communicator world;
  std::vector<char> input = {};
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataParallel->inputs_count.emplace_back(input.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(wordCount.data()));
    taskDataParallel->outputs_count.emplace_back(wordCount.size());
  }

  TestTaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    ASSERT_EQ(wordCount[0], 0);
  }
}

// Тестирование стpоки с 3 словами
TEST(CountWordsMPI, TestThreeWords) {
  boost::mpi::communicator world;
  std::vector<char> input;
  std::string testString = "three funny words";
  for (char c : testString) {
    input.push_back(c);
  }
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataParallel->inputs_count.emplace_back(input.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(wordCount.data()));
    taskDataParallel->outputs_count.emplace_back(wordCount.size());
  }

  TestTaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    ASSERT_EQ(wordCount[0], 3);
  }
}

}  // namespace burykin_m_word_count