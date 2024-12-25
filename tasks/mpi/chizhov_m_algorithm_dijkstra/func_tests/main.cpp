// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iomanip>
#include <random>
#include <vector>

#include "mpi/chizhov_m_algorithm_dijkstra/include/ops_mpi.hpp"
void chizhov_m_dijkstra_mpi::generateMatrix(std::vector<int> &w, int n, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(min, max);
  for (int i = 0; i < n * n; i++) {
    int val = dist(gen);
    w[i] = val;
  }
  for (int i = 0; i < n; i++) {
    w[i * n + i] = 0;
  }
}

TEST(chizhov_m_dijkstra_realization_mpi, Test_Graph_5_vertex) {
  boost::mpi::communicator world;
  int size = 5;
  int st = 0;
  int min = 1;
  int max = 10;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size, min, max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  chizhov_m_dijkstra_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(st);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    chizhov_m_dijkstra_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res);
  }
}

TEST(chizhov_m_dijkstra_realization_mpi, Test_Graph_10_vertex) {
  boost::mpi::communicator world;
  int size = 10;
  int st = 3;
  int min = 5;
  int max = 50;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size, min, max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  chizhov_m_dijkstra_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(st);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    chizhov_m_dijkstra_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res);
  }
}

TEST(chizhov_m_dijkstra_realization_mpi, Test_Graph_13_vertex) {
  boost::mpi::communicator world;
  int size = 13;
  int st = 3;
  int min = 4;
  int max = 20;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size, min, max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  chizhov_m_dijkstra_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(st);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    chizhov_m_dijkstra_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res);
  }
}

TEST(chizhov_m_dijkstra_realization_mpi, Test_Graph_20_vertex) {
  boost::mpi::communicator world;
  int size = 20;
  int st = 3;
  int min = 2;
  int max = 40;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size, min, max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  chizhov_m_dijkstra_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(st);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    chizhov_m_dijkstra_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res);
  }
}

TEST(chizhov_m_dijkstra_realization_mpi, Test_Source_Vertex_False) {
  boost::mpi::communicator world;
  int size = 10;
  int st = 13;
  int min = 2;
  int max = 20;
  std::vector<int> matrix(size * size, 0);
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size, min, max);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  chizhov_m_dijkstra_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(chizhov_m_dijkstra_realization_mpi, Test_Negative_Value) {
  boost::mpi::communicator world;
  int size = 3;
  int st = 0;
  std::vector<int> matrix = {0, 2, 5, 4, 0, 2, 3, -1, 0};
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  chizhov_m_dijkstra_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(chizhov_m_dijkstra_realization_mpi, Test_Spare_Graph_5x5) {
  boost::mpi::communicator world;
  int size = 5;
  int st = 0;
  // Create data
  std::vector<int> matrix = {0, 5, 0, 3, 0, 0, 0, 4, 2, 2, 0, 0, 0, 3, 0, 0, 3, 0, 0, 2, 9, 0, 1, 0, 0};
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(size);
    taskDataPar->inputs_count.emplace_back(st);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  chizhov_m_dijkstra_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> res_seq(size, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(size);
    taskDataSeq->inputs_count.emplace_back(st);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    // Create Task
    chizhov_m_dijkstra_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(res_seq, res);
  }
}