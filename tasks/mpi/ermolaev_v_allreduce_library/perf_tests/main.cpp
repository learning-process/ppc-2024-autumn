// Copyright 2023 Nesterov Alexander
#include <vector>

#include "mpi/ermolaev_v_allreduce_library/include/ops_mpi.hpp"
#include "mpi/ermolaev_v_allreduce_library/include/test_funcs.hpp"

template <typename _T>
using MPIAllReduce = ermolaev_v_allreduce_mpi::DefaultAllReduceTask<_T>;

TEST(ermolaev_v_allreduce_mpi, test_pipeline_run_mpi_allreduce) {
  ermolaev_v_allreduce_mpi::perfTestBody<MPIAllReduce<double>, double>(2500, 2500, ppc::core::PerfResults::PIPELINE);
}

TEST(ermolaev_v_allreduce_mpi, test_task_run_mpi_allreduce) {
  ermolaev_v_allreduce_mpi::perfTestBody<MPIAllReduce<double>, double>(2500, 2500, ppc::core::PerfResults::TASK_RUN);
}