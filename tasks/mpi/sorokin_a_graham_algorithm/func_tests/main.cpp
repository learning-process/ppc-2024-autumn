// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/sorokin_a_graham_algorithm/include/ops_mpi.hpp"

std::vector<int> getrndvec(int n, int max, int min) {
  std::uniform_real_distribution<double> unif(static_cast<double>(min), static_cast<double>(max));
  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());
  std::vector<int> tmp(n);
  for (int i = 0; i < n; i++) {
    tmp[i] = unif(rand_engine);
  }
  return tmp;
}

TEST(sorokin_a_graham_algorithm_MPI, Test_20_points) {
  boost::mpi::communicator world;
  std::vector<int> in = {12, 5, 2, 3,  5, 2, 7, 1, 8,  6, 3, 8, 6, 5, 9, 3, 4, 7, 10, 10,
                         2,  9, 5, 12, 0, 0, 8, 0, 10, 7, 6, 9, 1, 5, 3, 2, 2, 8, 7,  10};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {0, 0, 8, 0, 12, 5, 10, 10, 5, 12, 2, 9, 1, 5};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  sorokin_a_graham_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    sorokin_a_graham_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 0; i < outres.size(); i++) {
      ASSERT_EQ(outres[i], out[i]);
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}
TEST(sorokin_a_graham_algorithm_MPI, Test_10_points) {
  boost::mpi::communicator world;
  std::vector<int> in = {1, 4, 2, 8, 6, 4, 9, 3, 7, 6, 2, 2, 5, 1, 4, 9, 10, 10, 8, 2};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {5, 1, 8, 2, 9, 3, 10, 10, 4, 9, 2, 8, 1, 4, 2, 2};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  sorokin_a_graham_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    sorokin_a_graham_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 0; i < outres.size(); i++) {
      ASSERT_EQ(outres[i], out[i]);
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}
TEST(sorokin_a_graham_algorithm_MPI, Test_6_points) {
  boost::mpi::communicator world;
  std::vector<int> in = {-1, 0, 0, -1, -1, -1, -2, -1, -1, -2, -1, 0};
  std::vector<int> out(in.size(), 0);
  std::vector<int> outres = {-1, -2, 0, -1, -1, 0, -2, -1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  sorokin_a_graham_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    sorokin_a_graham_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 0; i < outres.size(); i++) {
      ASSERT_EQ(outres[i], out[i]);
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}
TEST(sorokin_a_graham_algorithm_MPI, Test_rnd_100000_points) {
  boost::mpi::communicator world;
  std::vector<int> in = getrndvec(200000, -100, 100);
  std::vector<int> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  sorokin_a_graham_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    sorokin_a_graham_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}
TEST(sorokin_a_graham_algorithm_MPI, Test_rnd_1000000_points) {
  boost::mpi::communicator world;
  std::vector<int> in = getrndvec(2000000, -100, 100);
  std::vector<int> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  sorokin_a_graham_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    sorokin_a_graham_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 0; i < out.size(); i++) {
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}
