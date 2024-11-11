#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/chernova_n_word_count/include/ops_mpi.hpp"

std::vector<char> generateWords(int k) {
  const std::string words[] = {"one", "two", "three"};

  std::string result;
  int j = words->size();

  for (int i = 0; i < k; ++i) {
    result += words[rand() % (j)];
    if (i < k - 1) {
      result += ' ';
    }
  }

  return std::vector<char>(result.begin(), result.end());
}
int k = 50;
std::vector<char> testDataParallel = generateWords(k);

TEST(chernova_n_word_count_mpi, Test_empty_string) {
  boost::mpi::communicator world;
  std::vector<char> in = {};
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }
  chernova_n_word_count_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataSequential->inputs_count.emplace_back(in.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    chernova_n_word_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], referenceWordCount[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_five_words) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string testString = "This is a test phrase";
  in.resize(testString.size());
  std::copy(testString.begin(), testString.end(), in.begin());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataSequential->inputs_count.emplace_back(in.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    chernova_n_word_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], referenceWordCount[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_five_words_with_space_and_hyphen) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string testString = "This   is a - test phrase";
  in.resize(testString.size());
  std::copy(testString.begin(), testString.end(), in.begin());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataSequential->inputs_count.emplace_back(in.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    chernova_n_word_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], referenceWordCount[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_ten_words) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string testString = "This is a test phrase, I really love this phrase";
  in.resize(testString.size());
  std::copy(testString.begin(), testString.end(), in.begin());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataSequential->inputs_count.emplace_back(in.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    chernova_n_word_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], referenceWordCount[0]);
  }
}
TEST(chernova_n_word_count_mpi, Test_five_words_with_a_lot_of_space) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string testString = "This               is           a             test                phrase";
  in.resize(testString.size());
  std::copy(testString.begin(), testString.end(), in.begin());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataSequential->inputs_count.emplace_back(in.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    chernova_n_word_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], referenceWordCount[0]);
  }
}
TEST(chernova_n_word_count_mpi, Test_twenty_words) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string testString =
      "This is a test phrase, I really love this phrase. This is a test phrase, I really love this phrase";
  in.resize(testString.size());
  std::copy(testString.begin(), testString.end(), in.begin());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataSequential->inputs_count.emplace_back(in.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    chernova_n_word_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], referenceWordCount[0]);
  }
}

TEST(chernova_n_word_count_mpi, Test_five_words_with_space_in_the_end) {
  boost::mpi::communicator world;
  std::vector<char> in;
  std::string testString = "This is a test phrase           ";
  in.resize(testString.size());
  std::copy(testString.begin(), testString.end(), in.begin());
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataSequential->inputs_count.emplace_back(in.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    chernova_n_word_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], referenceWordCount[0]);
  }
}

TEST(Sequential_chernova_n_word_count, Test_random_fifty_words) {
  boost::mpi::communicator world;
  std::vector<char> in = testDataParallel;
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataParallel->inputs_count.emplace_back(in.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  chernova_n_word_count_mpi::TestMPITaskParallel testTaskParallel(taskDataParallel);
  ASSERT_TRUE(testTaskParallel.validation());
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> referenceWordCount(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<char *>(in.data())));
    taskDataSequential->inputs_count.emplace_back(in.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(referenceWordCount.data()));
    taskDataSequential->outputs_count.emplace_back(referenceWordCount.size());

    chernova_n_word_count_mpi::TestMPITaskSequential testTaskSequential(taskDataSequential);
    ASSERT_TRUE(testTaskSequential.validation());
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], referenceWordCount[0]);
  }
}