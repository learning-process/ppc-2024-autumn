// Copyright 2024 Tselikova Arina
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/tselikova_a_average_of_vector_elements/include/ops_mpi.hpp"

TEST(tselikova_a_average_of_vector_elements_mpi, Test_Average_Simple) {
  boost::mpi::communicator world;
  std::vector<int> global_vec{4, 6, 8};
  std::vector<float> global_avg{0.0f};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_avg.data()));
    taskDataPar->outputs_count.emplace_back(global_avg.size());
  }

  tselikova_a_average_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    float reference_avg = 6.0f;

    ASSERT_FLOAT_EQ(global_avg[0], reference_avg);
  }
}

TEST(tselikova_a_average_of_vector_elements_mpi, Test_Average_LargeVector) {
  boost::mpi::communicator world;
  std::vector<int> large_vec(1000, 1);
  std::vector<float> global_avg{0.0f};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(large_vec.data()));
    taskDataPar->inputs_count.emplace_back(large_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_avg.data()));
    taskDataPar->outputs_count.emplace_back(global_avg.size());
  }

  tselikova_a_average_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    float reference_avg = 1.0f;

    ASSERT_FLOAT_EQ(global_avg[0], reference_avg);
  }
}

TEST(tselikova_a_average_of_vector_elements_mpi, Test_EmptyVector) {
  boost::mpi::communicator world;
  std::vector<int> empty_vec;
  std::vector<float> global_avg{0.0f};

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(empty_vec.data()));
    taskDataPar->inputs_count.emplace_back(empty_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_avg.data()));
    taskDataPar->outputs_count.emplace_back(global_avg.size());
  }

  tselikova_a_average_of_vector_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}