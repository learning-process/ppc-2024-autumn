// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shishkarev_a_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;

  if (world.rank() == 0) {
    count_size_vector = 100000000;  // Создаем вектор из 100 миллионов единиц
    global_vec = std::vector<int>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto MPIVectorSumParallel =
      std::make_shared<shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel>(taskDataPar);
  ASSERT_TRUE(MPIVectorSumParallel->validation());
  MPIVectorSumParallel->pre_processing();
  MPIVectorSumParallel->run();
  MPIVectorSumParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], count_size_vector);
  }
}

TEST(shishkarev_a_sum_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_size_vector;

  if (world.rank() == 0) {
    count_size_vector = 100000000;  // Создаем вектор из 100 миллионов единиц
    global_vec = std::vector<int>(count_size_vector, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto MPIVectorSumParallel =
      std::make_shared<shishkarev_a_sum_of_vector_elements_mpi::MPIVectorSumParallel>(taskDataPar);
  ASSERT_TRUE(MPIVectorSumParallel->validation());
  MPIVectorSumParallel->pre_processing();
  MPIVectorSumParallel->run();
  MPIVectorSumParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], count_size_vector);
  }
}
