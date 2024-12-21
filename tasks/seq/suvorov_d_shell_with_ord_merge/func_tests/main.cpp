// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>
#include <algorithm>

#include "seq/suvorov_d_shell_with_ord_merge/include/ops_seq.hpp"

TEST(suvorov_d_shell_with_ord_merge_seq, Sorting_a_vector_with_unique_elements) {
  std::vector<int> data_to_sort = {5, 2, 7, 4, 1, 3, 9};
  size_t count_of_elems = data_to_sort.size();
  std::vector<int> sorted_result(count_of_elems, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingSeq = std::make_shared<ppc::core::TaskData>();
  taskDataForSortingSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(data_to_sort.data()));
  taskDataForSortingSeq->inputs_count.emplace_back(data_to_sort.size());
  taskDataForSortingSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sorted_result.data()));
  taskDataForSortingSeq->outputs_count.emplace_back(sorted_result.size());

  suvorov_d_shell_with_ord_merge_seq::TaskShellSortSeq ShellSortSeq(taskDataForSortingSeq);
  ASSERT_EQ(ShellSortSeq.validation(), true);
  ShellSortSeq.pre_processing();
  ShellSortSeq.run();
  ShellSortSeq.post_processing();
  EXPECT_TRUE(std::is_sorted(sorted_result.begin(), sorted_result.end()));
}
