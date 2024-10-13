#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kurakin_m_min_values_by_rows_matrix/include/ops_mpi.hpp"

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand1) {
  const int count_rows = 10;
  const int size_rows = 12;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_min_vec(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                         size_rows);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, par_min_vec);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand2) {
  const int count_rows = 10;
  const int size_rows = 13;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_min_vec(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                         size_rows);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, par_min_vec);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand3) {
  const int count_rows = 10;
  const int size_rows = 13;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_min_vec(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                         size_rows);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, par_min_vec);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand4) {
  const int count_rows = 10;
  const int size_rows = 15;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_min_vec(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                         size_rows);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, par_min_vec);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand5) {
  const int count_rows = 10;
  const int size_rows = 2;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_min_vec(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                         size_rows);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, par_min_vec);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_validation) {
  const int count_rows = 10;
  const int size_rows = 0;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, count_rows, size_rows);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> ref_min_vec(count_rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, count_rows,
                                                                                         size_rows);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
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
