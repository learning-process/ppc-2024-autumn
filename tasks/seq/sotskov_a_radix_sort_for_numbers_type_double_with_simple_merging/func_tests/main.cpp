#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "seq/sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging/include/ops_seq.hpp"

namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi {
std::vector<double> generate_random_input_with_same_integer_part(size_t size, int integer_part = 1,
                                                                 double min_fraction = 0.0, double max_fraction = 1.0) {
  std::vector<double> input(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(min_fraction, max_fraction);

  for (size_t i = 0; i < size; ++i) {
    input[i] = integer_part + dis(gen);
  }
  std::sort(input.rbegin(), input.rend());
  return input;
}

std::vector<double> run_sort(const std::vector<double> &input) {
  std::vector<double> output(input.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(const_cast<double *>(input.data())));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq::TestTaskSequential testTaskSequential(
      taskDataSeq);

  EXPECT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  return output;
}
}  // namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, RandomSortedSameIntegerPart) {
  auto input = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::
      generate_random_input_with_same_integer_part(5, 1, 0.0, 0.99);
  auto output = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::run_sort(input);

  std::vector<double> expected = input;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(output, expected);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, EmptyArray) {
  std::vector<double> input = {};
  auto output = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::run_sort(input);
  ASSERT_EQ(output, input);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, SingleElement) {
  std::vector<double> input = {42.42};
  auto output = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::run_sort(input);
  ASSERT_EQ(output, input);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, AlreadySorted) {
  std::vector<double> input = {1.1, 2.2, 3.3, 4.4, 5.5};
  auto output = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::run_sort(input);
  ASSERT_EQ(output, input);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, ReverseSorted) {
  std::vector<double> input = {5.5, 4.4, 3.3, 2.2, 1.1};
  auto output = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::run_sort(input);

  std::vector<double> expected = input;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(output, expected);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, LargeArray) {
  size_t size = 100000;
  auto input = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::
      generate_random_input_with_same_integer_part(size, 1, 0.0, 0.99);
  auto output = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::run_sort(input);

  std::vector<double> expected = input;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(output, expected);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, NegativeNumbers) {
  std::vector<double> input = {-1.1, -3.3, -2.2, -4.4, -5.5};
  auto output = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::run_sort(input);

  std::vector<double> expected = input;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(output, expected);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_seq, MixedNegativePositive) {
  std::vector<double> input = {1.1, -2.2, 3.3, -4.4, 5.5};
  auto output = sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::run_sort(input);

  std::vector<double> expected = input;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(output, expected);
}
