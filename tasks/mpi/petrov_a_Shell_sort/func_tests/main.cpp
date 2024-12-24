#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <random>
#include <vector>

#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

namespace petrov_a_Shell_sort_mpi {

std::vector<int> generate_random_vector(int n, int min_val = -100, int max_val = 100,
                                        unsigned seed = std::random_device{}()) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(min_val, max_val);

  std::vector<int> vec(n);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
  return vec;
}

void template_test(const std::vector<int>& input_data) {
  boost::mpi::communicator world;
  std::vector<int> data;
  std::vector<int> result_data;

  int vector_size = static_cast<int>(input_data.size());
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data = input_data;
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskData->inputs_count.emplace_back(data.size());
    result_data.resize(vector_size);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskData->outputs_count.emplace_back(result_data.size());
  }

  auto taskMPI = std::make_shared<TestTaskMPI>(taskData);

  ASSERT_TRUE(taskMPI->validation());
  ASSERT_TRUE(taskMPI->pre_processing());
  taskMPI->run();
  taskMPI->post_processing();

  if (world.rank() == 0) {
    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}

}  // namespace petrov_a_Shell_sort_mpi

TEST(petrov_a_Shell_sort_mpi, test_sorted_ascending) {
  petrov_a_Shell_sort_mpi::template_test({1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(petrov_a_Shell_sort_mpi, test_sorted_descending) {
  petrov_a_Shell_sort_mpi::template_test({8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(petrov_a_Shell_sort_mpi, test_almost_sorted_random) {
  petrov_a_Shell_sort_mpi::template_test({9, 7, 5, 3, 1, 2, 4, 6});
}

TEST(petrov_a_Shell_sort_mpi, test_all_equal_elements) {
  petrov_a_Shell_sort_mpi::template_test({5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(petrov_a_Shell_sort_mpi, test_random_vector) {
  auto random_vec = petrov_a_Shell_sort_mpi::generate_random_vector(1000, -1000, 1000);
  petrov_a_Shell_sort_mpi::template_test(random_vec);
}

TEST(petrov_a_Shell_sort_mpi, test_large_random_vector) {
  auto large_random_vec = petrov_a_Shell_sort_mpi::generate_random_vector(100000, -10000, 10000);
  petrov_a_Shell_sort_mpi::template_test(large_random_vec);
}