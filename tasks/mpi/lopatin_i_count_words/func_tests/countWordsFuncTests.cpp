#include <gtest/gtest.h>

#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

TEST(lopatin_i_count_words_mpi, test_empty_string) {
  boost::mpi::communicator world;
  std::vector<char> input = {};
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataParallel->inputs_count.emplace_back(input.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataParallel->outputs_count.emplace_back(wordCount.size());

    lopatin_i_count_words_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
    ASSERT_FALSE(testTaskParallel.validation());
  }
}

TEST(lopatin_i_count_words_mpi, test_1_word) {
  boost::mpi::communicator world;
  std::vector<char> input;
  std::string testString = "sym";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    input.push_back(testString[j]);
  }
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataParallel->inputs_count.emplace_back(input.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataParallel->outputs_count.emplace_back(wordCount.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    lopatin_i_count_words_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], referenceWordCount[0]);
  }
}

TEST(lopatin_i_count_words_mpi, test_3_words) {
  boost::mpi::communicator world;
  std::vector<char> input;
  std::string testString = "three funny words";
  for (unsigned long int j = 0; j < testString.length(); j++) {
    input.push_back(testString[j]);
  }
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataParallel->inputs_count.emplace_back(input.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataParallel->outputs_count.emplace_back(wordCount.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    lopatin_i_count_words_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], referenceWordCount[0]);
  }
}

TEST(lopatin_i_count_words_mpi, test_300_words) {
  boost::mpi::communicator world;
  std::vector<char> input = lopatin_i_count_words_mpi::generateLongString(20);
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataParallel->inputs_count.emplace_back(input.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataParallel->outputs_count.emplace_back(wordCount.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    lopatin_i_count_words_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], referenceWordCount[0]);
  }
}

TEST(lopatin_i_count_words_mpi, test_1500_words) {
  boost::mpi::communicator world;
  std::vector<char> input = lopatin_i_count_words_mpi::generateLongString(100);
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataParallel->inputs_count.emplace_back(input.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataParallel->outputs_count.emplace_back(wordCount.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    lopatin_i_count_words_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], referenceWordCount[0]);
  }
}

TEST(lopatin_i_count_words_mpi, test_6k_words) {
  boost::mpi::communicator world;
  std::vector<char> input = lopatin_i_count_words_mpi::generateLongString(400);
  std::vector<int> wordCount(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataParallel->inputs_count.emplace_back(input.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataParallel->outputs_count.emplace_back(wordCount.size());
  }

  lopatin_i_count_words_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(input.data())));
    taskDataSequential->inputs_count.emplace_back(input.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    lopatin_i_count_words_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], referenceWordCount[0]);
  }
}