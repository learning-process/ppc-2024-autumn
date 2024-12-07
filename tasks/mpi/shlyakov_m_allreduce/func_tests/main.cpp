// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>

#include "mpi/shlyakov_m_allreduce/include/ops_mpi.hpp"

TEST(shlyakov_m_all_reduce_mpi, allreduce_test_with_matrix_10_10) {
  int row = 10;
  int col = 10;

  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(shlyakov_m_all_reduce_mpi, my_allreduce_test_with_matrix_10_10) {
  int row = 10;
  int col = 10;

  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(shlyakov_m_all_reduce_mpi, allreduce_test_with_matrix_10_12) {
  int row = 10;
  int col = 12;

  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(shlyakov_m_all_reduce_mpi, my_allreduce_test_with_matrix_10_12) {
  int row = 10;
  int col = 12;

  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(shlyakov_m_all_reduce_mpi, allreduce_test_with_matrix_10_15) {
  int row = 10;
  int col = 15;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(shlyakov_m_all_reduce_mpi, my_allreduce_test_with_matrix_10_15) {
  int row = 10;
  int col = 15;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(shlyakov_m_all_reduce_mpi, allreduce_test_with_matrix_10_2) {
  int row = 10;
  int col = 2;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(shlyakov_m_all_reduce_mpi, my_allreduce_test_with_matrix_10_2) {
  int row = 10;
  int col = 2;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, m_vec);
  }
}

TEST(shlyakov_m_all_reduce_mpi, allreduce_test_with_matrix_0_0) {
  int row = 0;
  int col = 0;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(shlyakov_m_all_reduce_mpi, my_allreduce_test_with_matrix_0_0) {
  int row = 0;
  int col = 0;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> m_vec(row, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = shlyakov_m_all_reduce_mpi::TestMPITaskSequential::generate_matrix(row, col);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(shlyakov_m_all_reduce_mpi, allreduce_test_0_0_1) {
  int row = 3;
  int col = 6;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> ans;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {10, 7, 4, 8, 7, 9, 13, 4, 5, 7, 6, 9, 12, 4, 2, 5, 3, 9};
    ans = {0, 0, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, ans);
    ASSERT_EQ(m_vec, ans);
  }
}

TEST(shlyakov_m_all_reduce_mpi, my_allreduce_test_0_0_1) {
  int row = 3;
  int col = 6;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> ans;
  std::vector<int32_t> m_vec(row);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = {10, 7, 4, 8, 7, 9, 13, 4, 5, 7, 6, 9, 12, 4, 2, 5, 3, 9};
    ans = {0, 0, 1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(m_vec.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  shlyakov_m_all_reduce_mpi::MyTestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> rm_vec(row);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(row);
    taskDataSeq->inputs_count.emplace_back(col);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(rm_vec.data()));
    taskDataSeq->outputs_count.emplace_back(row);

    shlyakov_m_all_reduce_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(rm_vec, ans);
    ASSERT_EQ(m_vec, ans);
  }
}