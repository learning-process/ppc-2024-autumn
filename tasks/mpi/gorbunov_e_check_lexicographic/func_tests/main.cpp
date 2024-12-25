#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "mpi/gorbunov_e_check_lexicographic/include/ops_mpi.hpp"

TEST(gorbunov_e_check_lexicographic_mpi, difference_on_fourth_symbol) {
  boost::mpi::communicator world;
  std::vector<char> global_word_left = {'H', 'e', 'l', 'l', 'o'};
  std::vector<char> global_word_right = {'H', 'e', 'l', 'p'};
  std::vector<int32_t> parallel_result(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_left.data()));
    taskDataPar->inputs_count.emplace_back(global_word_left.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_right.data()));
    taskDataPar->inputs_count.emplace_back(global_word_right.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    taskDataPar->outputs_count.emplace_back(parallel_result.size());
  }

  gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> seq_result(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_left.data()));
    taskDataSeq->inputs_count.emplace_back(global_word_left.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_right.data()));
    taskDataPar->inputs_count.emplace_back(global_word_right.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    // Create Task
    gorbunov_e_check_lexicographic_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(seq_result[0], parallel_result[0]);
  }
}


TEST(gorbunov_e_check_lexicographic_mpi, difference_on_length) {
  boost::mpi::communicator world;
  std::vector<char> global_word_left = {'H', 'e', 'l', 'l', 'o'};
  std::vector<char> global_word_right = {'H', 'e', 'l', 'l', 'o', '!'};
  std::vector<int32_t> parallel_result(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_left.data()));
    taskDataPar->inputs_count.emplace_back(global_word_left.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_right.data()));
    taskDataPar->inputs_count.emplace_back(global_word_right.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    taskDataPar->outputs_count.emplace_back(parallel_result.size());
  }

  gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> seq_result(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_left.data()));
    taskDataSeq->inputs_count.emplace_back(global_word_left.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_right.data()));
    taskDataPar->inputs_count.emplace_back(global_word_right.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    // Create Task
    gorbunov_e_check_lexicographic_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(seq_result[0], parallel_result[0]);
  }
}

TEST(gorbunov_e_check_lexicographic_mpi, no_difference) {
  boost::mpi::communicator world;
  std::vector<char> global_word_left = {'H', 'e', 'l', 'l', 'o'};
  std::vector<char> global_word_right = {'H', 'e', 'l', 'l', 'o'};
  std::vector<int32_t> parallel_result(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_left.data()));
    taskDataPar->inputs_count.emplace_back(global_word_left.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_right.data()));
    taskDataPar->inputs_count.emplace_back(global_word_right.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    taskDataPar->outputs_count.emplace_back(parallel_result.size());
  }

  gorbunov_e_check_lexicographic_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> seq_result(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_left.data()));
    taskDataSeq->inputs_count.emplace_back(global_word_left.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_word_right.data()));
    taskDataPar->inputs_count.emplace_back(global_word_right.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    // Create Task
    gorbunov_e_check_lexicographic_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(seq_result[0], parallel_result[0]);
  }
}