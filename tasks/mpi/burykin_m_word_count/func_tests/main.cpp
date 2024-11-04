#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <string>
#include <vector>

#include "mpi/burykin_m_word_count/include/ops_mpi.hpp"

std::vector<char> burykin_m_word_count::RandomSentence(int size) {
  std::vector<char> vec(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size > 0) {
    vec[size - 1] = 0x61 + gen() % 26;
    vec[0] = 0x41 + gen() % 26;
  }
  for (int i = 1; i < size - 1; i++) {
    if (vec[i - 1] != ' ' && gen() % 4 == 0) {
      vec[i] = ' ';
    } else {
      vec[i] = 0x61 + gen() % 26;
    }
  }
  return vec;
}

TEST(burykin_m_word_count_MPI_func, TestEmptyString) {
  int length = 0;

  // Create data
  std::vector<char> input(length);
  std::vector<int> wordCount(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(input.size());
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataPar->outputs_count.emplace_back(wordCount.size());
  }

  // Create Task
  burykin_m_word_count::TestTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.size());

    // Create Task
    burykin_m_word_count::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], local_count[0]);
  }
}

TEST(burykin_m_word_count_MPI_func, TestStringABC) {
  std::string input_str = "abc";
  std::vector<char> input(input_str.begin(), input_str.end());
  std::vector<int> wordCount(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(input.size());
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataPar->outputs_count.emplace_back(wordCount.size());
  }

  // Create Task
  burykin_m_word_count::TestTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.size());

    // Create Task
    burykin_m_word_count::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], local_count[0]);
  }
}

TEST(burykin_m_word_count_MPI_func, TestLength30) {
  int length = 30;

  // Create data
  std::vector<char> input(length);
  std::vector<int> wordCount(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(input.size());
  if (world.rank() == 0) {
    input = burykin_m_word_count::RandomSentence(length);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataPar->outputs_count.emplace_back(wordCount.size());
  }

  // Create Task
  burykin_m_word_count::TestTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.size());

    // Create Task
    burykin_m_word_count::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], local_count[0]);
  }
}

TEST(burykin_m_word_count_MPI_func, TestLength50) {
  int length = 50;

  // Create data
  std::vector<char> input(length);
  std::vector<int> wordCount(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(input.size());
  if (world.rank() == 0) {
    input = burykin_m_word_count::RandomSentence(length);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataPar->outputs_count.emplace_back(wordCount.size());
  }

  // Create Task
  burykin_m_word_count::TestTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.size());

    // Create Task
    burykin_m_word_count::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], local_count[0]);
  }
}

TEST(burykin_m_word_count_MPI_func, TestLength99) {
  int length = 99;

  // Create data
  std::vector<char> input(length);
  std::vector<int> wordCount(1, 0);

  boost::mpi::communicator world;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(input.size());
  if (world.rank() == 0) {
    input = burykin_m_word_count::RandomSentence(length);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(wordCount.data()));
    taskDataPar->outputs_count.emplace_back(wordCount.size());
  }

  // Create Task
  burykin_m_word_count::TestTaskParallel testTaskParallel(taskDataPar);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> local_count(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(local_count.data()));
    taskDataSeq->outputs_count.emplace_back(local_count.size());

    // Create Task
    burykin_m_word_count::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(wordCount[0], local_count[0]);
  }
}
