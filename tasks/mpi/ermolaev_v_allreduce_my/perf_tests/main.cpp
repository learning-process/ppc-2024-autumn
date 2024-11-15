// Copyright 2023 Nesterov Alexander
#include <vector>

#include "mpi/ermolaev_v_allreduce_my/include/ops_mpi.hpp"
#include "mpi/ermolaev_v_allreduce_my/include/test_funcs.hpp"

template <typename _T>
using MyAllReduce = ermolaev_v_allreduce_mpi::MyAllReduceTask<_T>;

TEST(ermolaev_v_allreduce_mpi, test_pipeline_run_my_allreduce) {
  ermolaev_v_allreduce_mpi::perfTestBody<MyAllReduce<double>, double>(2500, 2500, ppc::core::PerfResults::PIPELINE);
}

TEST(ermolaev_v_allreduce_mpi, test_task_run_my_allreduce) {
  ermolaev_v_allreduce_mpi::perfTestBody<MyAllReduce<double>, double>(2500, 2500, ppc::core::PerfResults::TASK_RUN);
}
