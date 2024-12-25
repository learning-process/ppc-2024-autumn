#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/task/include/task.hpp"
#include "mpi/kazunin_n_quicksort_simple_merge/include/ops_mpi.hpp"

namespace kazunin_n_quicksort_simple_merge_mpi {

std::vector<int> generate_random_vector(int n, int min_val = -100, int max_val = 100,
                                        unsigned seed = std::random_device{}()) {
  static std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(min_val, max_val);

  std::vector<int> vec(n);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
  return vec;
}

void template_test(int vector_size) {
  boost::mpi::communicator world;
  std::vector<int> data;
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    data = generate_random_vector(vector_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataPar->inputs_count.emplace_back(data.size());

    result_data.resize(vector_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  auto taskParallel = std::make_shared<QuicksortSimpleMerge>(taskDataPar);

  bool success = taskParallel->validation();
  boost::mpi::broadcast(world, success, 0);
  if (success) {
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::sort(data.begin(), data.end());
      EXPECT_EQ(data, result_data);
    }
  }
}

void template_test(const std::vector<int>& input_data) {
  boost::mpi::communicator world;
  std::vector<int> data = input_data;
  std::vector<int> result_data;

  int vector_size = static_cast<int>(data.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataPar->inputs_count.emplace_back(data.size());

    result_data.resize(vector_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
    taskDataPar->outputs_count.emplace_back(result_data.size());
  }

  auto taskParallel = std::make_shared<QuicksortSimpleMerge>(taskDataPar);

  bool success = taskParallel->validation();
  boost::mpi::broadcast(world, success, 0);
  if (success) {
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::sort(data.begin(), data.end());
      EXPECT_EQ(data, result_data);
    }
  }
}

}  // namespace kazunin_n_quicksort_simple_merge_mpi

TEST(kazunin_n_quicksort_simple_merge_mpi, test_sorted_ascending) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({1, 2, 3, 4, 5, 6, 8, 9, 5, 4, 3, 2, 1});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_almost_sorted_random) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({9, 7, 5, 3, 1, 2, 4, 6, 8, 10});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_sorted_descending) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_all_equal_elements) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_negative_numbers) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({0, -1, -2, -3, -4, -5, -6, -7, -8, -9});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_mixed_order) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({1, 3, 2, 4, 6, 5, 7, 9, 8, 10});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_single_element) {
  std::vector<int> vec = {42};
  kazunin_n_quicksort_simple_merge_mpi::template_test(vec);
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_empty_vector) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_mixed_large_and_small_numbers) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({100, 99, 98, 1, 2, 3, 4, 5});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_positive_and_negative_numbers) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({-5, -10, 5, 10, 0});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_large_random_numbers) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({123, 456, 789, 321, 654, 987});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_random_numbers) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({9, 1, 4, 7, 2, 8, 5, 3, 6});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_max_and_min_int) {
  kazunin_n_quicksort_simple_merge_mpi::template_test(
      {std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_prime_numbers) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({11, 13, 17, 19, 23, 29, 31});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_descending_with_negatives) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({50, 40, 30, 20, 10, 0, -10, -20, -30});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_pi_digits) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_consecutive_mixed_numbers) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({12, 14, 16, 18, 20, 15, 17, 19, 21});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_descending_with_zero_and_negatives) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({7, 6, 5, 4, 3, 2, 1, 0, -1, -2});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_large_integers) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({1000, 2000, 1500, 2500, 1750});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_mixed_digits_and_zero) {
  kazunin_n_quicksort_simple_merge_mpi::template_test({8, 4, 2, 6, 1, 9, 5, 3, 7, 0});
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_random_generated_10_elements) {
  kazunin_n_quicksort_simple_merge_mpi::template_test(10);
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_random_generated_20_elements) {
  kazunin_n_quicksort_simple_merge_mpi::template_test(20);
}

TEST(kazunin_n_quicksort_simple_merge_mpi, test_random_generated_23_elements) {
  kazunin_n_quicksort_simple_merge_mpi::template_test(23);
}
