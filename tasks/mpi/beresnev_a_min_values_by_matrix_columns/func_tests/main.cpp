// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/beresnev_a_min_values_by_matrix_columns/include/ops_mpi.hpp"

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Empty_Input_0) {
  boost::mpi::communicator world;
  const int N = 0;
  const int M = 3;

  std::vector<int> in(N * M, 0);
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Empty_Input_1) {
  boost::mpi::communicator world;
  const int N = 6;
  const int M = 0;

  std::vector<int> in{};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Wrong_Size_0) {
  boost::mpi::communicator world;
  const int N = -2;
  const int M = 3;

  std::vector<int> in{10, 1, 2, -1, -100, 2};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Wrong_Size_1) {
  boost::mpi::communicator world;
  const int N = 2;
  const int M = 312;

  std::vector<int> in{10, 1, 2, -1, -100, 2};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Test_Identity_Matrix) {
  boost::mpi::communicator world;
  const int N = 1;
  const int M = 1;

  std::vector<int> in{10};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);
  std::vector<int> gold{10};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(gold, out);
  }
}
TEST(beresnev_a_min_values_by_matrix_columns_mpi, Test_Base_0) {
  boost::mpi::communicator world;
  const int N = 2;
  const int M = 3;

  std::vector<int> in{10, 1, 2, -1, -100, 2};
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);
  const std::vector<int> gold{-1, -100, 2};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(gold, out);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Test_Base_1) {
  boost::mpi::communicator world;
  const int N = 100;
  const int M = 100;

  std::vector<int> in;
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = beresnev_a_min_values_by_matrix_columns_mpi::getRandomVector(N * M);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(M, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataSeq->inputs_count.emplace_back(n.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataSeq->inputs_count.emplace_back(m.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference));
    taskDataSeq->outputs_count.emplace_back(out.size());
    // Create Task
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference, out);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Test_Base_2) {
  boost::mpi::communicator world;
  const int N = 43;
  const int M = 563;

  std::vector<int> in;
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = beresnev_a_min_values_by_matrix_columns_mpi::getRandomVector(N * M);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(M, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataSeq->inputs_count.emplace_back(n.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataSeq->inputs_count.emplace_back(m.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference));
    taskDataSeq->outputs_count.emplace_back(out.size());
    // Create Task
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference, out);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Test_Base_3) {
  boost::mpi::communicator world;
  const int N = 908;
  const int M = 510;

  std::vector<int> in;
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = beresnev_a_min_values_by_matrix_columns_mpi::getRandomVector(N * M);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(M, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataSeq->inputs_count.emplace_back(n.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataSeq->inputs_count.emplace_back(m.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference));
    taskDataSeq->outputs_count.emplace_back(out.size());
    // Create Task
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference, out);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Test_Base_4) {
  boost::mpi::communicator world;
  const int N = 1;
  const int M = 1000;

  std::vector<int> in;
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = beresnev_a_min_values_by_matrix_columns_mpi::getRandomVector(N * M);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(M, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataSeq->inputs_count.emplace_back(n.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataSeq->inputs_count.emplace_back(m.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference));
    taskDataSeq->outputs_count.emplace_back(out.size());
    // Create Task
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference, out);
  }
}

TEST(beresnev_a_min_values_by_matrix_columns_mpi, Test_Base_5) {
  boost::mpi::communicator world;
  const int N = 1000;
  const int M = 1000;

  std::vector<int> in;
  std::vector<int> out(M, 0);
  std::vector<int> n(1, N);
  std::vector<int> m(1, M);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = beresnev_a_min_values_by_matrix_columns_mpi::getRandomVector(N * M);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataPar->inputs_count.emplace_back(n.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataPar->inputs_count.emplace_back(m.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference(M, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(n.data()));
    taskDataSeq->inputs_count.emplace_back(n.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(m.data()));
    taskDataSeq->inputs_count.emplace_back(m.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference));
    taskDataSeq->outputs_count.emplace_back(out.size());
    // Create Task
    beresnev_a_min_values_by_matrix_columns_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference, out);
  }
}