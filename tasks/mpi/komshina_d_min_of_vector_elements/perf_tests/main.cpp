#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/komshina_d_min_of_vector_elements/include/ops_mpi.hpp"

TEST(komshina_d_min_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_min(1, 0);
  int expected_min_value;

  // Создание TaskData
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 100000000;  
    const int start = -50000000; 
    expected_min_value = start;   
    global_vec.resize(count);
    for (int i = 0; i < count; ++i) {
      global_vec[i] = start + i;  
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation()); 
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_min_value, global_min[0]);  
  }
}

TEST(komshina_d_min_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_min(1, 0);
  int expected_min_value;

  // Создание TaskData
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 100000000;  
    const int start = -50000000;  
    expected_min_value = start;   
    global_vec.resize(count);
    for (int i = 0; i < count; ++i) {
      global_vec[i] = start + i;  
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<komshina_d_min_of_vector_elements_mpi::MinOfVectorElementsTaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());  
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(expected_min_value, global_min[0]);  
  }
}

