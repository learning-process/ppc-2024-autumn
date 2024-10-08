#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kurakin_m_min_values_by_rows_matrix/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, Test_Min_Rand) {
  const int count_rows = 10;
  const int size_rows = 12;
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<std::vector<int32_t>> global_sum(count_rows, std::vector<int32_t>(1, 0));
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    for (int i = 0; i < count_rows; i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum[i].data()));
      taskDataPar->outputs_count.emplace_back(global_sum[i].size());
    }
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<std::vector<int32_t>> reference_sum(count_rows, std::vector<int32_t>(1, 0));

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    for (int i = 0; i < count_rows; i++) {
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum[i].data()));
      taskDataSeq->outputs_count.emplace_back(reference_sum[i].size());
    }

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                         size_rows);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum, global_sum);
  }
}

TEST(Parallel_Operations_MPI, Test_Min) {
  const int count_rows = 2;
  const int size_rows = 9;
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<std::vector<int32_t>> global_sum(count_rows, std::vector<int32_t>(1, 0));
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = std::vector<int>{9, 8, 7, 6, 5, 4, 3, 2, 1, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    for (int i = 0; i < count_rows; i++) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum[i].data()));
      taskDataPar->outputs_count.emplace_back(global_sum[i].size());
    }
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<std::vector<int32_t>> reference_sum(count_rows, std::vector<int32_t>(1, 0));

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    for (int i = 0; i < count_rows; i++) {
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum[i].data()));
      taskDataSeq->outputs_count.emplace_back(reference_sum[i].size());
    }

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                         size_rows);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_sum, global_sum);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
