// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/suvorov_d_shell_with_ord_merge/include/ops_mpi.hpp"

namespace suvorov_d_shell_with_ord_merge_mpi {
void check_correctness_cases(const std::vector<int> &correct_test_data) {
  boost::mpi::communicator world;
  std::vector<int> data_to_sort;
  size_t count_of_elems = 0;
  std::vector<int> sorted_result_mpi;
  std::shared_ptr<ppc::core::TaskData> taskDataForSortingMpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data_to_sort = correct_test_data;
    count_of_elems = data_to_sort.size();
    sorted_result_mpi.assign(count_of_elems, 0);

    taskDataForSortingMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
    taskDataForSortingMpi->inputs_count.emplace_back(data_to_sort.size());
    taskDataForSortingMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result_mpi.data()));
    taskDataForSortingMpi->outputs_count.emplace_back(sorted_result_mpi.size());
  }

  suvorov_d_shell_with_ord_merge_mpi::TaskShellSortParallel ShellSortMpi(taskDataForSortingMpi);
  ASSERT_EQ(ShellSortMpi.validation(), true);
  ShellSortMpi.pre_processing();
  ShellSortMpi.run();
  ShellSortMpi.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> sorted_result_seq(count_of_elems, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
    taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
    taskDataForSortingSeq->inputs_count.emplace_back(data_to_sort.size());
    taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result_seq.data()));
    taskDataForSortingSeq->outputs_count.emplace_back(sorted_result_seq.size());

    suvorov_d_shell_with_ord_merge_mpi::TaskShellSortSeq ShellSortSeq(taskDataForSortingSeq);
    ASSERT_EQ(ShellSortSeq.validation(), true);
    ShellSortSeq.pre_processing();
    ShellSortSeq.run();
    ShellSortSeq.post_processing();

    bool test_result = sorted_result_seq == sorted_result_mpi &&
                       std::is_sorted(sorted_result_seq.begin(), sorted_result_seq.end()) &&
                       std::is_sorted(sorted_result_mpi.begin(), sorted_result_mpi.end());
    EXPECT_TRUE(test_result);
  }
}
}  // namespace suvorov_d_shell_with_ord_merge_mpi

TEST(suvorov_d_shell_with_ord_merge_mpi, Sorting_a_vector_with_unique_elements) {
  suvorov_d_shell_with_ord_merge_mpi::check_correctness_cases({5, 2, 7, 4, 1, 3, 9});
}

TEST(suvorov_d_shell_with_ord_merge_mpi, Sorting_a_vector_with_repeating_elements) {
  suvorov_d_shell_with_ord_merge_mpi::check_correctness_cases({5, 2, 4, 8, 5, 1, 2, 2, 5, 9, 8});
}

TEST(suvorov_d_shell_with_ord_merge_mpi, Sorting_a_vector_with_a_single_element) {
  suvorov_d_shell_with_ord_merge_mpi::check_correctness_cases({31});
}

TEST(suvorov_d_shell_with_ord_merge_mpi, Sorting_a_vector_with_negative_elements) {
  suvorov_d_shell_with_ord_merge_mpi::check_correctness_cases({5, -2, 4, 7, -12, 10, 3, -5});
}

TEST(suvorov_d_shell_with_ord_merge_mpi, Sorting_an_empty_vector) {
  boost::mpi::communicator world;
  std::vector<int> data_to_sort;
  size_t count_of_elems;
  std::vector<int> sorted_result_mpi;
  std::shared_ptr<ppc::core::TaskData> taskDataForSortingMpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data_to_sort = {};
    count_of_elems = data_to_sort.size();
    sorted_result_mpi.assign(count_of_elems, 0);

    taskDataForSortingMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
    taskDataForSortingMpi->inputs_count.emplace_back(data_to_sort.size());
    taskDataForSortingMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result_mpi.data()));
    taskDataForSortingMpi->outputs_count.emplace_back(sorted_result_mpi.size());
  }

  suvorov_d_shell_with_ord_merge_mpi::TaskShellSortParallel ShellSortMpi(taskDataForSortingMpi);
  EXPECT_FALSE(ShellSortMpi.validation());
}
