#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kovalchuk_a_odd_even_megre_sort/include/ops_mpi.hpp"

using namespace kovalchuk_a_odd_even;

std::vector<int> getRandomVector(int sz, int min = -999, int max = 999);

std::vector<int> getRandomVector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

TEST(kovalchuk_a_odd_even, Test_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int> global_result(12, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vector = {8, 2, 5, 10, 1, 7, 3, 12, 6, 11, 4, 9};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  kovalchuk_a_odd_even::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_result = global_vector;
    std::sort(reference_result.begin(), reference_result.end());

    ASSERT_EQ(reference_result, global_result);
  }
}

TEST(kovalchuk_a_odd_even, Test_Vector_10) {
  const int count_elements = 10;
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int> global_result(count_elements, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vector = getRandomVector(count_elements);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(count_elements);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  kovalchuk_a_odd_even::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_result = global_vector;
    std::sort(reference_result.begin(), reference_result.end());

    ASSERT_EQ(reference_result, global_result);
  }
}

TEST(kovalchuk_a_odd_even, Test_Vector_1) {
  const int count_elements = 1;
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int> global_result(count_elements, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vector = getRandomVector(count_elements, 0, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(count_elements);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  kovalchuk_a_odd_even::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_result = global_vector;
    std::sort(reference_result.begin(), reference_result.end());

    ASSERT_EQ(reference_result, global_result);
  }
}