#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "seq/sarafanov_m_quick_sort_batcher_merge/include/ops_seq.hpp"

namespace sarafanov_m_quick_sort_batcher_merge_seq {

std::vector<int> generate_random_vector(int n, int min_val = -100, int max_val = 100,
                                        unsigned seed = std::random_device{}()) {
  static std::mt19937 gen(seed);
  std::uniform_int_distribution<int> dist(min_val, max_val);

  std::vector<int> vec(n);
  std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });
  return vec;
}

void template_test(const std::vector<int>& input_data) {
  std::vector<int> data = input_data;
  std::vector<int> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataPar->inputs_count.emplace_back(data.size());

  result_data.resize(data.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskDataPar->outputs_count.emplace_back(result_data.size());

  auto taskParallel = std::make_shared<QuicksortBatcherMerge>(taskDataPar);

  if (taskParallel->validation()) {
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}

}  // namespace sarafanov_m_quick_sort_batcher_merge_seq

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_sorted_ascending) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_almost_sorted_random) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({9, 7, 5, 3, 1, 2, 4, 6});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_sorted_descending) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_all_equal_elements) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({5, 5, 5, 5, 5, 5, 5, 5});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_negative_numbers) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({0, -1, -2, -3, -4, -5, -6, -7});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_mixed_order) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({1, 3, 2, 4, 5, 7, 6, 8});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_single_element) {
  std::vector<int> vec = {42};
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(vec);
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_empty_vector) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_mixed_large_and_small_numbers) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({100, 99, 50, 25, 12, 6, 3, 1});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_positive_and_negative_numbers) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({-5, -10, 5, 10, 0});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_large_random_numbers) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({512, 256, 128, 64, 32, 16, 8, 4});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_random_numbers) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({7, 1, 4, 2, 8, 6, 5, 3});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_max_and_min_int) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(
      {std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_prime_numbers) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({2, 3, 5, 7, 11, 13, 17, 19});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_descending_with_negatives) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({64, 32, 16, 8, 4, 2, 0, -2});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_pi_digits) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({3, 1, 4, 1, 5, 9, 2, 6});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_consecutive_mixed_numbers) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({12, 14, 16, 18, 15, 17, 19, 21});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_descending_with_zero_and_negatives) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({7, 6, 5, 4, 3, 2, 1, 0});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_large_integers) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({1024, 512, 256, 128, 64});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_mixed_digits_and_zero) {
  sarafanov_m_quick_sort_batcher_merge_seq::template_test({8, 4, 2, 6, 1, 9, 5, 3});
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_random_vector_size_4) {
  auto vec = sarafanov_m_quick_sort_batcher_merge_seq::generate_random_vector(4);
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(vec);
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_random_vector_size_8) {
  auto vec = sarafanov_m_quick_sort_batcher_merge_seq::generate_random_vector(8);
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(vec);
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_random_vector_size_16) {
  auto vec = sarafanov_m_quick_sort_batcher_merge_seq::generate_random_vector(16);
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(vec);
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_random_vector_size_32) {
  auto vec = sarafanov_m_quick_sort_batcher_merge_seq::generate_random_vector(32);
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(vec);
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_random_vector_size_64) {
  auto vec = sarafanov_m_quick_sort_batcher_merge_seq::generate_random_vector(64);
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(vec);
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_random_vector_size_128) {
  auto vec = sarafanov_m_quick_sort_batcher_merge_seq::generate_random_vector(128);
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(vec);
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, test_random_vector_size_256) {
  auto vec = sarafanov_m_quick_sort_batcher_merge_seq::generate_random_vector(256);
  sarafanov_m_quick_sort_batcher_merge_seq::template_test(vec);
}
