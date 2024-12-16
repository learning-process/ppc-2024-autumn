#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
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
  std::vector<int> data = input_data;
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskData->inputs_count.emplace_back(data.size());

  result_data.resize(data.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskData->outputs_count.emplace_back(result_data.size());

  auto taskMPI = std::make_shared<TestTaskMPI>(taskData);

  if (taskMPI->validation()) {
    if (taskMPI->pre_processing()) {
      taskMPI->run();
      taskMPI->post_processing();
    } else {
      return;
    }

    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}

TEST(petrov_a_Shell_sort_mpi, test_sorted_ascending_mpi) {
  petrov_a_Shell_sort_mpi::template_test({1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(petrov_a_Shell_sort_mpi, test_almost_sorted_random_mpi) {
  petrov_a_Shell_sort_mpi::template_test({9, 7, 5, 3, 1, 2, 4, 6});
}

TEST(petrov_a_Shell_sort_mpi, test_sorted_descending_mpi) {
  petrov_a_Shell_sort_mpi::template_test({8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(petrov_a_Shell_sort_mpi, test_all_equal_elements_mpi) {
  petrov_a_Shell_sort_mpi::template_test({5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(petrov_a_Shell_sort_mpi, test_random_vector_mpi) {
  auto random_vec = generate_random_vector(1000, -1000, 1000);
  petrov_a_Shell_sort_mpi::template_test(random_vec);
}

}  // namespace petrov_a_Shell_sort_mpi
