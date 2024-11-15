// Copyright 2023 Nesterov Alexander
#include <vector>

#include "seq/ermolaev_v_allreduce/include/ops_seq.hpp"
#include "seq/ermolaev_v_allreduce/include/test_funcs.hpp"

TEST(ermolaev_v_allreduce_seq, test_pipeline_run) {
  ermolaev_v_allreduce_seq::perfTestBody<ermolaev_v_allreduce_seq::TestTaskSequential<double>, double>(
      2500, 2500, ppc::core::PerfResults::PIPELINE);
}

TEST(ermolaev_v_allreduce_seq, test_task_run) {
  ermolaev_v_allreduce_seq::perfTestBody<ermolaev_v_allreduce_seq::TestTaskSequential<double>, double>(
      2500, 2500, ppc::core::PerfResults::TASK_RUN);
}