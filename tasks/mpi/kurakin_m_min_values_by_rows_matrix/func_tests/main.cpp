#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kurakin_m_min_values_by_rows_matrix/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, Test_Min_Rand1) {
  int count_rows = 10;
  //int size_rows = 20;
  for (int size_rows = 4; size_rows < 54; size_rows+=7) {
    boost::mpi::communicator world;
    std::vector<int> global_vec;
    std::vector<int32_t> global_sum(count_rows, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      global_vec = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
      taskDataPar->inputs_count.emplace_back(global_vec.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
      taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows,
                                                                                     size_rows);
    EXPECT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      // Create data
      std::vector<int32_t> reference_sum(count_rows, 0);

      // Create TaskData
      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
      taskDataSeq->inputs_count.emplace_back(global_vec.size());
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
      taskDataSeq->outputs_count.emplace_back(reference_sum.size());

      // Create Task
      kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                           size_rows);
      EXPECT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();

      EXPECT_EQ(reference_sum, global_sum);
      std::cout << "count_rows: " << count_rows << "; size_rows: " << size_rows << "\n";
    }
  }
}

TEST(Parallel_Operations_MPI, Test_Min_Rand2) {
  const int count_rows = 300;
  const int size_rows = 2;
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_sum(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

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
