// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/naumov_b_bubble_sort/include/ops_mpi.hpp"

TEST(naumov_b_bubble_sort, test_pipeline_run) {
  boost::mpi::communicator world;
  int rank = world.rank();
  std::vector<int> global_vec;
  std::vector<int> global_out;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  const size_t count_size_vector = 1200;
  if (rank == 0) {
    global_vec.resize(count_size_vector);
    global_out.resize(count_size_vector);

    std::srand(static_cast<int>(std::time(nullptr)) + rank);
    std::generate(global_vec.begin(), global_vec.end(), []() { return std::rand() % 10000 - 5000; });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  auto testMpiParallel = std::make_shared<naumov_b_bubble_sort_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_TRUE(testMpiParallel->validation());
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();

  world.barrier();

  if (rank == 0) {
    std::vector<int> expected_vec = global_vec;
    std::sort(expected_vec.begin(), expected_vec.end());

    ASSERT_TRUE(std::equal(expected_vec.begin(), expected_vec.end(), global_out.begin()))
        << "Sorted results do not match.";
  }
}

TEST(naumov_b_bubble_sort, test_task_run) {
  boost::mpi::communicator world;
  int rank = world.rank();
  std::vector<int> global_vec;
  std::vector<int> global_out;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  const size_t count_size_vector = 2000;
  if (rank == 0) {
    global_vec.resize(count_size_vector);
    global_out.resize(count_size_vector);

    std::srand(static_cast<int>(std::time(nullptr)) + rank);
    std::generate(global_vec.begin(), global_vec.end(), []() { return std::rand() % 10000 - 5000; });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  auto testMpiParallel = std::make_shared<naumov_b_bubble_sort_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_TRUE(testMpiParallel->validation());
  testMpiParallel->pre_processing();
  testMpiParallel->run();
  testMpiParallel->post_processing();

  world.barrier();

  if (rank == 0) {
    std::vector<int> expected_vec = global_vec;
    std::sort(expected_vec.begin(), expected_vec.end());

    ASSERT_TRUE(std::equal(expected_vec.begin(), expected_vec.end(), global_out.begin()))
        << "Sorted results do not match.";
  }
}
