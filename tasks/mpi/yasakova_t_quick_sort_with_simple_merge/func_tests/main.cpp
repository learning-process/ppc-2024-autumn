#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/task/include/task.hpp"
#include "mpi/yasakova_t_quick_sort_with_simple_merge/include/ops_mpi.hpp"

namespace yasakova_t_quick_sort_with_simple_merge_mpi {

std::vector<int> create_random_integer_vector(int size, int minimum_value = -100, int maximum_value = 100,
                                              unsigned random_seed = std::random_device{}()) {
  static std::mt19937 generator(random_seed);
  std::uniform_int_distribution<int> distribution(minimum_value, maximum_value);

  std::vector<int> random_vector(size);
  std::generate(random_vector.begin(), random_vector.end(), [&]() { return distribution(generator); });
  return random_vector;
}

void execute_parallel_sort_test(int vector_length) {
  boost::mpi::communicator mpi_comm;
  std::vector<int> input_vector;
  std::vector<int> output_vector;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_length));
  task_data->inputs_count.emplace_back(1);

  if (mpi_comm.rank() == 0) {
    input_vector = create_random_integer_vector(vector_length);
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());

    output_vector.resize(vector_length);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  auto parallel_sort_task =
      std::make_shared<yasakova_t_quick_sort_with_simple_merge_mpi::SimpleMergeQuicksort>(task_data);

  bool is_valid = parallel_sort_task->validation();
  boost::mpi::broadcast(mpi_comm, is_valid, 0);
  if (is_valid) {
    parallel_sort_task->pre_processing();
    parallel_sort_task->run();
    parallel_sort_task->post_processing();

    if (mpi_comm.rank() == 0) {
      std::sort(input_vector.begin(), input_vector.end());
      EXPECT_EQ(input_vector, output_vector);
    }
  }
}

void execute_parallel_sort_test(const std::vector<int>& input_vector) {
  boost::mpi::communicator mpi_comm;
  std::vector<int> local_data = input_vector;
  std::vector<int> sorted_data;

  int vector_size = static_cast<int>(local_data.size());
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size));
  task_data->inputs_count.emplace_back(1);

  if (mpi_comm.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(local_data.data()));
    task_data->inputs_count.emplace_back(local_data.size());

    sorted_data.resize(vector_size);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_data.data()));
    task_data->outputs_count.emplace_back(sorted_data.size());
  }

  auto parallel_sort_task =
      std::make_shared<yasakova_t_quick_sort_with_simple_merge_mpi::SimpleMergeQuicksort>(task_data);

  bool is_valid = parallel_sort_task->validation();
  boost::mpi::broadcast(mpi_comm, is_valid, 0);
  if (is_valid) {
    parallel_sort_task->pre_processing();
    parallel_sort_task->run();
    parallel_sort_task->post_processing();

    if (mpi_comm.rank() == 0) {
      std::sort(local_data.begin(), local_data.end());
      EXPECT_EQ(local_data, sorted_data);
    }
  }
}
}  // namespace yasakova_t_quick_sort_with_simple_merge_mpi

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_sorted_array_ascending) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({1, 2, 3, 4, 5, 6, 8, 9, 5, 4, 3, 2, 1});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_almost_sorted_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({9, 7, 5, 3, 1, 2, 4, 6, 8, 10});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_sorted_array_descending) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_array_with_equal_elements) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_array_with_negative_numbers) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({0, -1, -2, -3, -4, -5, -6, -7, -8, -9});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_mixed_order_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({1, 3, 2, 4, 6, 5, 7, 9, 8, 10});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_single_element_array) {
  std::vector<int> vec = {42};
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(vec);
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_empty_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_mixed_large_and_small_numbers_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({100, 99, 98, 1, 2, 3, 4, 5});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_positive_and_negative_numbers_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({-5, -10, 5, 10, 0});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_large_random_numbers_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({123, 456, 789, 321, 654, 987});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_random_numbers_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({9, 1, 4, 7, 2, 8, 5, 3, 6});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_max_and_min_integer_values) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(
      {std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_descending_array_with_negatives) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({50, 40, 30, 20, 10, 0, -10, -20, -30});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_pi_digits_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_consecutive_mixed_numbers_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({12, 14, 16, 18, 20, 15, 17, 19, 21});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_descending_with_zero_and_negatives_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({7, 6, 5, 4, 3, 2, 1, 0, -1, -2});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_mixed_digits_and_zero_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({8, 4, 2, 6, 1, 9, 5, 3, 7, 0});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_random_generated_array_with_10_elements) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(10);
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_random_generated_array_with_20_elements) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(20);
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_random_generated_array_with_23_elements) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test(23);
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_prime_numbers_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({11, 13, 17, 19, 23, 29, 31});
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_large_integers_array) {
  yasakova_t_quick_sort_with_simple_merge_mpi::execute_parallel_sort_test({1000, 2000, 1500, 2500, 1750});
}