// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/vladimirova_j_gather/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, vladimirova_j_gather_first_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = 
  {2,2,-1,2,2,2,2,2,-1,2,2,2,-1,2,2,2,-1,-1,2,2,2,1,2,2,2,1,2,2,2,2,2,1,2,2,2,2,1,2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> global_max(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    std::for_each(global_vector.begin(), global_vector.end(), [](int number) {
      std::cout << number << " "; 
    });
    std::cout << " GGGGGGGGGGGGGGGGG \n";
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  int otv =1;
  for (int i = 1; i < world.size(); i++) otv *= i;
  vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  ASSERT_EQ(otv, global_max[0]);
  
}


TEST(Parallel_Operations_MPI, vladimirova_j_gather_second_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> global_max(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  global_vector.push_back(3);
  global_vector.push_back(3);
  global_vector.push_back(3);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  int otv = 1;
  for (int i = 1; i < world.size(); i++) otv *= i;
  vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  ASSERT_EQ(otv, global_max[0]);
}
