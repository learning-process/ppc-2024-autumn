
#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "mpi/strakhov_a_str_char_freq/include/ops_mpi.hpp"

TEST(strakhov_a_str_char_freq_mpi, simple_test) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<int32_t> global_count(1, 0);
  char target = '2';
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_str = {'1', '2', '3'};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }
  strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference_count(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_count.data()));
    taskDataSeq->outputs_count.emplace_back(reference_count.size());
    strakhov_a_str_char_freq_mpi::StringCharactersFrequencySequentional testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_count[0], global_count[0]);
  }
}
TEST(strakhov_a_str_char_freq_mpi, test_full) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<int32_t> global_count(1, 0);
  char target = '1';
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_str = 240;
    global_str = std::vector<char>(count_size_str, '1');
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }
  strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  ASSERT_EQ(global_count[0], 1);
}

TEST(strakhov_a_str_char_freq_mpi, test_one_half) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<int32_t> global_count(1, 0);
  char target = '2';
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_str = 240;
    global_str = std::vector<char>(count_size_str, '1');
    for (int i = 0; i < count_size_str; i += 2) {
      global_str[i] = '2';
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }
  strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  ASSERT_EQ(global_count[0], 0.5);
}

TEST(strakhov_a_str_char_freq_mpi, test_empty) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<int32_t> global_count(1, 0);
  char target = '2';
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_str = 240;
    global_str = std::vector<char>(count_size_str, '1');
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }
  strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  ASSERT_EQ(global_count[0], 0);
}


TEST(strakhov_a_str_char_freq_mpi, test_null) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<int32_t> global_count(1, 0);
  char target = '1';
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }
  strakhov_a_str_char_freq_mpi::StringCharactersFrequencyParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(global_count[0], 0);
  }
}
