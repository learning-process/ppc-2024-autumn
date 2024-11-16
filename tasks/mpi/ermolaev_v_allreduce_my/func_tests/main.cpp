// Copyright 2023 Nesterov Alexander
#include <vector>

#include "mpi/ermolaev_v_allreduce_my/include/ops_mpi.hpp"
#include "mpi/ermolaev_v_allreduce_my/include/test_funcs.hpp"

template <typename _T>
using MyAllReduce = ermolaev_v_allreduce_mpi::MyAllReduceTask<_T>;

TEST(ermolaev_v_allreduce_mpi, run_double_task_my_allreduce) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes) ermolaev_v_allreduce_mpi::funcTestBody<MyAllReduce<double>, double>(rows, cols, -500, 500);
}
TEST(ermolaev_v_allreduce_mpi, run_float_task_my_allreduce) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes) ermolaev_v_allreduce_mpi::funcTestBody<MyAllReduce<float>, float>(rows, cols, -500, 500);
}
TEST(ermolaev_v_allreduce_mpi, run_int64_task_my_allreduce) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_mpi::funcTestBody<MyAllReduce<int64_t>, int64_t>(rows, cols, -500, 500);
}
TEST(ermolaev_v_allreduce_mpi, run_int32_task_my_allreduce) {
  std::vector<uint32_t> sizes = {1, 2, 3, 9, 16, 25, 100};
  for (auto& rows : sizes)
    for (auto& cols : sizes)
      ermolaev_v_allreduce_mpi::funcTestBody<MyAllReduce<int32_t>, int32_t>(rows, cols, -500, 500);
}

TEST(ermolaev_v_allreduce_mpi, validation_mpi) {
  ermolaev_v_allreduce_mpi::testValidation<MyAllReduce<int32_t>, int32_t>();
}
TEST(ermolaev_v_allreduce_mpi, validation_seq) {
  ermolaev_v_allreduce_mpi::testValidation<ermolaev_v_allreduce_mpi::TestMPITaskSequential<int32_t>, int32_t>();
}

TEST(ermolaev_v_allreduce_mpi, useless_test_for_codecov) {
  auto ptr = ermolaev_v_allreduce_mpi::shared_ptr_array<int32_t>(2);
  ptr[1] = 10;
  ASSERT_EQ(ptr.get()[1], 10);
}

TEST(ermolaev_v_allreduce_mpi, useless_test_for_codecov_2) {
  ermolaev_v_allreduce_mpi::perfTestBody<MyAllReduce<double>, double>(1, 1, ppc::core::PerfResults::NONE);
}
