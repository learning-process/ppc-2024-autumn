#include <gtest/gtest.h>

#include "seq/tyurin_m_shell_sort_batcher_merge/include/ops_seq.hpp"

namespace tyurin_m_shell_sort_batcher_merge_seq {

void run_test_template(const std::vector<int>& data, const int n) {
  std::vector<int> input_vec = data;
  std::vector<int> output_vec(n);

  auto task = std::make_shared<ppc::core::TaskData>();

  task->inputs_count.emplace_back(n);
  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vec.data()));

  ShellSortBatcherMerge test(task);
  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  std::vector<int> exp_result = input_vec;
  std::sort(exp_result.begin(), exp_result.end());
  EXPECT_EQ(exp_result, output_vec);
}

void validation_test_template(const int n) {
  std::vector<int> input_vec;
  std::vector<int> output_vec;

  auto task = std::make_shared<ppc::core::TaskData>();
  task->inputs_count.emplace_back(n);
  ShellSortBatcherMerge test(task);

  ASSERT_FALSE(test.validation());
}

}  // namespace tyurin_m_shell_sort_batcher_merge_seq

TEST(tyurin_m_shell_sort_batcher_merge_seq, all_permutations_test) {
  std::vector<int> data = {-4, -3, -2, -1, 0, 1, 2, 3};
  do {
    tyurin_m_shell_sort_batcher_merge_seq::run_test_template(data, data.size());
  } while (std::next_permutation(data.begin(), data.end()));
}

TEST(tyurin_m_shell_sort_batcher_merge_seq, validation_test) {
  tyurin_m_shell_sort_batcher_merge_seq::validation_test_template(0);
  tyurin_m_shell_sort_batcher_merge_seq::validation_test_template(3);
}
