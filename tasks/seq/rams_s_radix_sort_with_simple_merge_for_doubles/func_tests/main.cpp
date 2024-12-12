#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "seq/rams_s_radix_sort_with_simple_merge_for_doubles/include/ops_seq.hpp"

class rams_s_radix_sort_with_simple_merge_for_doubles_seq_test : public ::testing::TestWithParam<std::vector<double>> {
};

TEST_P(rams_s_radix_sort_with_simple_merge_for_doubles_seq_test, p) {
  auto in = GetParam();
  std::vector<double> out(GetParam().size(), 0);
  std::vector<double> expected(in);
  std::sort(expected.begin(), expected.end());

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data->inputs_count.emplace_back(in.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  rams_s_radix_sort_with_simple_merge_for_doubles_seq::TaskSequential test_task(task_data);
  ASSERT_EQ(test_task.validation(), true);
  test_task.pre_processing();
  test_task.run();
  test_task.post_processing();
  ASSERT_EQ(expected, out);
}

std::vector<double> rams_s_radix_sort_with_simple_merge_for_doubles_mpi_test_gen_input(size_t length) {
  std::vector<double> vec(length, 0);
  std::random_device dev;
  std::mt19937_64 gen(dev());
  for (size_t i = 0; i < length; i++) {
    while (std::isnan(vec[i] = std::bit_cast<double>(gen())));
  }
  return vec;
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
  rams_s_radix_sort_with_simple_merge_for_doubles_seq_test,
  rams_s_radix_sort_with_simple_merge_for_doubles_seq_test,
  ::testing::Values(
    std::vector<double>{},
    std::vector<double>{10.4, 5.8, -6.000008},
    std::vector<double>{10, 5, 6, 15},
    std::vector<double>{513, 512, 257, 256},
    std::vector<double>{10.06, 5.55, 6.87, 15.0908},
    std::vector<double>{8989, 0.0, -std::numeric_limits<double>::infinity(), -0.0, 4.0, std::numeric_limits<double>::infinity(), -5.0, -3.0},
    rams_s_radix_sort_with_simple_merge_for_doubles_mpi_test_gen_input(5),
    rams_s_radix_sort_with_simple_merge_for_doubles_mpi_test_gen_input(16),
    rams_s_radix_sort_with_simple_merge_for_doubles_mpi_test_gen_input(23),
    rams_s_radix_sort_with_simple_merge_for_doubles_mpi_test_gen_input(99),
    rams_s_radix_sort_with_simple_merge_for_doubles_mpi_test_gen_input(55),
    rams_s_radix_sort_with_simple_merge_for_doubles_mpi_test_gen_input(64),
    rams_s_radix_sort_with_simple_merge_for_doubles_mpi_test_gen_input(48)
  )
);
// clang-format on
