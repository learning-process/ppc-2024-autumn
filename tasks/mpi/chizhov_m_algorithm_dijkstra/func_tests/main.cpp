// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/chizhov_m_algorithm_dijkstra/include/ops_mpi.hpp"
void chizhov_m_dijkstra_mpi::generateMatrix(std::vector<std::vector<int>> &w, int n) {
  srand(time(0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        w[i][j] = 0;
      } else {
        int val = rand() % 10 + 1;
        w[i][j] = val;
        w[j][i] = val;
      }
    }
  }
}

TEST(chizhov_m_dijkstra_realization_mpi, Test_Graph_5_vertex) {
  boost::mpi::communicator world;
  int size = 5;
  int st = 0;

  std::vector<std::vector<int>> matrix;

  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size);
    for (int i = 0; i < size; i++) {
      matrix[i].resize(size);
    }
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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

  std::vector<std::vector<int>> matrix;
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size);
    for (int i = 0; i < size; i++) {
      matrix[i].resize(size);
    }
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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
  int size = 10;
  int st = 5;

  std::vector<std::vector<int>> matrix;
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size);
    for (int i = 0; i < size; i++) {
      matrix[i].resize(size);
    }
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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
  int st = 17;

  std::vector<std::vector<int>> matrix;
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size);
    for (int i = 0; i < size; i++) {
      matrix[i].resize(size);
    }
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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

  std::vector<std::vector<int>> matrix;
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix.resize(size);
    for (int i = 0; i < size; i++) {
      matrix[i].resize(size);
    }
    chizhov_m_dijkstra_mpi::generateMatrix(matrix, size);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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
  
  std::vector<std::vector<int>> matrix = {{0, 2, 5}, {4, 0, 2}, {3, -1, 0}};
  std::vector<int> res(size, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix[i].data()));
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