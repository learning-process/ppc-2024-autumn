// Copyright 2023 Nesterov Alexander
#include <vector>

#include "mpi/ermolaev_v_allreduce_library/include/ops_mpi.hpp"
#include "mpi/ermolaev_v_allreduce_library/include/test_funcs.hpp"

template <typename _T>
using MPIAllReduce = ermolaev_v_allreduce_mpi::DefaultAllReduceTask<_T>;

TEST(ermolaev_v_allreduce_mpi, run_double_task_MPI_allreduce) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_mpi::funcTestBody<MPIAllReduce<double>, double>(rows, cols, -500, 500);
}
TEST(ermolaev_v_allreduce_mpi, run_float_task_MPI_allreduce) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes) ermolaev_v_allreduce_mpi::funcTestBody<MPIAllReduce<float>, float>(rows, cols, -500, 500);
}
TEST(ermolaev_v_allreduce_mpi, run_int64_task_MPI_allreduce) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_mpi::funcTestBody<MPIAllReduce<int64_t>, int64_t>(rows, cols, -500, 500);
}
TEST(ermolaev_v_allreduce_mpi, run_int32_task_MPI_allreduce) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_mpi::funcTestBody<MPIAllReduce<int32_t>, int32_t>(rows, cols, -500, 500);
}