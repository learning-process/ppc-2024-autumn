#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

inline std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100, 100);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST(konkov_mpi_dining_philosophers_func_test, Test_Dining_Philosophers) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 120;
    global_vec = getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  konkov_i_task_dining_philosophers::DiningPhilosophersMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (!testMpiTaskParallel.validation()) {
    GTEST_SKIP() << "Validation failed, skipping test.";
  }

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_result[0], std::accumulate(global_vec.begin(), global_vec.end(), 0));
  }
}
