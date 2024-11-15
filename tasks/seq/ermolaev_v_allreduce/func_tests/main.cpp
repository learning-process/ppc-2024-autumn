// Copyright 2023 Nesterov Alexander
#include <vector>

#include "seq/ermolaev_v_allreduce/include/ops_seq.hpp"
#include "seq/ermolaev_v_allreduce/include/test_funcs.hpp"

TEST(ermolaev_v_allreduce_seq, run_double_task) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_seq::funcTestBody<ermolaev_v_allreduce_seq::TestTaskSequential<double>, double>(rows, cols,
                                                                                                           -500, 500);
}
TEST(ermolaev_v_allreduce_seq, run_float_task) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_seq::funcTestBody<ermolaev_v_allreduce_seq::TestTaskSequential<float>, float>(rows, cols,
                                                                                                         -500, 500);
}
TEST(ermolaev_v_allreduce_seq, run_int64_task) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_seq::funcTestBody<ermolaev_v_allreduce_seq::TestTaskSequential<int64_t>, int64_t>(rows, cols,
                                                                                                             -500, 500);
}
TEST(ermolaev_v_allreduce_seq, run_int32_task) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_seq::funcTestBody<ermolaev_v_allreduce_seq::TestTaskSequential<int32_t>, int32_t>(rows, cols,
                                                                                                             -500, 500);
}