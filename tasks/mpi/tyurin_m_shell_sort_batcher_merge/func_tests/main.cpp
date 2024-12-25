#include <gtest/gtest.h>

#include <vector>

#include "mpi/tyurin_m_shell_sort_batcher_merge/include/ops_mpi.hpp"

namespace tyurin_m_shell_sort_batcher_merge_mpi {

void run_test_template(boost::mpi::communicator& world, const std::vector<int>& data, const int n) {
  std::vector<int> input_vec;
  std::vector<int> output_vec;

  auto task = std::make_shared<ppc::core::TaskData>();

  task->inputs_count.emplace_back(n);
  if (world.rank() == 0) {
    input_vec = data;
    output_vec.resize(n);

    task->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
    task->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vec.data()));
  }

  ShellSortBatcherMerge test(task);
  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    std::vector<int> exp_result = input_vec;
    std::sort(exp_result.begin(), exp_result.end());
    EXPECT_EQ(exp_result, output_vec);
  }
}

void validation_test_template(boost::mpi::communicator& world, const int n) {
  std::vector<int> input_vec;
  std::vector<int> output_vec;

  auto task = std::make_shared<ppc::core::TaskData>();
  task->inputs_count.emplace_back(n);
  ShellSortBatcherMerge test(task);

  if (world.rank() == 0) {
    ASSERT_FALSE(test.validation());
  }
}

}  // namespace tyurin_m_shell_sort_batcher_merge_mpi

TEST(tyurin_m_shell_sort_batcher_merge_mpi, all_permutations_test) {
  boost::mpi::communicator world;
  std::vector<int> data = {-4, -3, -2, -1, 0, 1, 2, 3};
  if (world.size() % 2 != 0) GTEST_SKIP();
  do {
    tyurin_m_shell_sort_batcher_merge_mpi::run_test_template(world, data, data.size());
  } while (std::next_permutation(data.begin(), data.end()));
}

TEST(tyurin_m_shell_sort_batcher_merge_mpi, validation_test_0_size) {
  boost::mpi::communicator world;
  tyurin_m_shell_sort_batcher_merge_mpi::validation_test_template(world, 0);
}

TEST(tyurin_m_shell_sort_batcher_merge_mpi, validation_test_n_not_even_size) {
  boost::mpi::communicator world;
  tyurin_m_shell_sort_batcher_merge_mpi::validation_test_template(world, 3);
}

TEST(tyurin_m_shell_sort_batcher_merge_mpi, validation_test_world_size_not_even) {
  boost::mpi::communicator world;
  if (world.size() == 3) tyurin_m_shell_sort_batcher_merge_mpi::validation_test_template(world, 4);
}
