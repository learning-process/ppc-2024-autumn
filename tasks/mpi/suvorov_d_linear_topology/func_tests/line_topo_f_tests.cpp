// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/example/include/ops_mpi.hpp"

TEST(suvorov_d_linear_topology, Test_Correctness) {
  boost::mpi::communicator world;
  std::vector<int> initial_data;
  std::vector<int> result_data;
  const int count_size_vector;
  
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_size_vector = 120;
    initial_data = suvorov_d_linear_topology::getRandomVector(count_size_vector);
    result_data.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(initial_data.data()));
    taskDataPar->inputs_count.emplace_back(initial_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  suvorov_d_linear_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int* order_data_ptr = reinterpret_cast<int*>(taskDataPar->outputs[1]);
    int order_data_size = taskDataPar->outputs_count[1];
    std::vector<int> order_vec(order_data_ptr, order_data_ptr + order_data_size);

    bool order_is_ok = true;
    for (int i = 0; i < world.size(); i++) {
      if (order_vec[i] != i) order_is_ok = false;
    }

    // Check correctness
    int result_data_size = taskDataPar->outputs_count[0];
    ASSERT_EQ(result_data_size, count_size_vector);
    ASSERT_EQ(initial_data, result_data);
    EXPECT_TRUE(order_is_ok);
  }
}
