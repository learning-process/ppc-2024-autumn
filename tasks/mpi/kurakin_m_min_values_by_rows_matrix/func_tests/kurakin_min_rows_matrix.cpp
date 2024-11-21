#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kurakin_m_min_values_by_rows_matrix/include/kurakin_min_rows_matrix_ops_mpi.hpp"

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_MPI_and_Seq_1_1) {
  int count_rows = 1;
  int size_rows = 1;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> ans;
  std::vector<int32_t> par_min_vec(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = {5};
    ans = {5};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, ans);
    ASSERT_EQ(par_min_vec, ans);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_MPI_and_Seq_5_1) {
  int count_rows = 5;
  int size_rows = 1;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> ans;
  std::vector<int32_t> par_min_vec(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = {5, 5, 5, 5, 5};
    ans = {5, 5, 5, 5, 5};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, ans);
    ASSERT_EQ(par_min_vec, ans);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_MPI_and_Seq_1_5) {
  int count_rows = 1;
  int size_rows = 5;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> ans;
  std::vector<int32_t> par_min_vec(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = {5, 5, 5, 5, 5};
    ans = {5};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, ans);
    ASSERT_EQ(par_min_vec, ans);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_MPI_and_Seq_3_5) {
  int count_rows = 3;
  int size_rows = 5;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> ans;
  std::vector<int32_t> par_min_vec(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = {1, 5, 3, 7, 9, 3, 4, 6, 7, 9, 2, 4, 2, 5, 0};
    ans = {1, 3, 0};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, ans);
    ASSERT_EQ(par_min_vec, ans);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_MPI_and_Seq_3_6) {
  int count_rows = 3;
  int size_rows = 6;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> ans;
  std::vector<int32_t> par_min_vec(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = {10, 5, 3, 9, 7, 9, 13, 4, 6, 7, 7, 9, 12, 4, 2, 5, 10, 9};
    ans = {3, 4, 2};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, ans);
    ASSERT_EQ(par_min_vec, ans);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_MPI_and_Seq_4_5) {
  int count_rows = 4;
  int size_rows = 5;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> ans;
  std::vector<int32_t> par_min_vec(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = {10, 5, 3, 9, 7, 9, 13, 4, 6, 7, 7, 9, 12, 4, 2, 5, 10, 9, 5, 8};
    ans = {3, 4, 2, 5};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, ans);
    ASSERT_EQ(par_min_vec, ans);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand_10_12) {
  int count_rows = 10;
  int size_rows = 12;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, par_min_vec);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand_10_15) {
  int count_rows = 10;
  int size_rows = 15;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, par_min_vec);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand_10_2) {
  int count_rows = 10;
  int size_rows = 2;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());
  }

  kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataSeq->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_min_vec.data()));
    taskDataSeq->outputs_count.emplace_back(ref_min_vec.size());

    // Create Task
    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_min_vec, par_min_vec);
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Min_Rand_0_0) {
  int count_rows = 0;
  int size_rows = 0;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> par_min_vec(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_mat = kurakin_m_min_values_by_rows_matrix_mpi::getRandomVector(count_rows * size_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&size_rows));
    taskDataPar->inputs_count.emplace_back(static_cast<size_t>(1));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(par_min_vec.data()));
    taskDataPar->outputs_count.emplace_back(par_min_vec.size());

    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(kurakin_m_min_values_by_rows_matrix_mpi, Test_Check_valdation) {
  int count_rows = 10;
  int size_rows = 10;
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

    kurakin_m_min_values_by_rows_matrix_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}