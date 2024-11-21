// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/poroshin_v_count_min_val_row_matrix_all_reduce/include/ops_mpi.hpp"

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, Test_rand_100_100) {
  int n = 100;
  int m = 100;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(m_vec, rm_vec);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_Test_rand_100_100) {
  int n = 100;
  int m = 100;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(m_vec, rm_vec);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, Test_2_0_1) {
  int n = 5;
  int m = 3;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> ans;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {0, 0, 6, 7, 4, 9, 4, 6, 7, 9, 3, 4, 8, 5, 0};
    ans = {2, 0, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(m_vec, ans);
    ASSERT_EQ(rm_vec, ans);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_Test_2_0_1) {
  int n = 5;
  int m = 3;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> ans;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {0, 0, 6, 7, 4, 9, 4, 6, 7, 9, 3, 4, 8, 5, 0};
    ans = {2, 0, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(m_vec, ans);
    ASSERT_EQ(rm_vec, ans);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, Test_0_0_1) {
  int m = 3;
  int n = 6;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> ans;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {10, 7, 4, 8, 7, 9, 13, 4, 5, 7, 6, 9, 12, 4, 2, 5, 3, 9};
    ans = {0, 0, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, ans);
    ASSERT_EQ(m_vec, ans);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_Test_0_0_1) {
  int m = 3;
  int n = 6;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> ans;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {10, 7, 4, 8, 7, 9, 13, 4, 5, 7, 6, 9, 12, 4, 2, 5, 3, 9};
    ans = {0, 0, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, ans);
    ASSERT_EQ(m_vec, ans);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, Test_0_0_1_1) {
  int m = 4;
  int n = 5;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> ans;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {9, 5, 3, 9, 7, 9, 13, 4, 5, 7, 7, 9, 12, 4, 0, 5, 11, 9, 0, 7};
    ans = {0, 0, 1, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, ans);
    ASSERT_EQ(m_vec, ans);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_Test_0_0_1_1) {
  int m = 4;
  int n = 5;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> ans;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {9, 5, 3, 9, 7, 9, 13, 4, 5, 7, 7, 9, 12, 4, 0, 5, 11, 9, 0, 7};
    ans = {0, 0, 1, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, ans);
    ASSERT_EQ(m_vec, ans);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, Test_rand_10_12) {
  int m = 10;
  int n = 12;

  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_Test_rand_10_12) {
  int m = 10;
  int n = 12;

  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, Test_rand_10_15) {
  int m = 10;
  int n = 15;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_Test_rand_10_15) {
  int m = 10;
  int n = 15;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, Test_rand_10_2) {
  int m = 10;
  int n = 2;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_Test_rand_10_2) {
  int m = 10;
  int n = 2;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);
  }

  poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> rm_vec(m, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(m);

    // Create Task
    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, Test_rand_0_0) {
  int m = 0;
  int n = 0;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);

    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(poroshin_v_count_min_val_row_matrix_all_reduce_mpi, My_Test_rand_0_0) {
  int m = 0;
  int n = 0;
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int32_t> m_vec(m, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = poroshin_v_count_min_val_row_matrix_all_reduce_mpi::TestMPITaskSequential::gen(m, n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(m);

    poroshin_v_count_min_val_row_matrix_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}