// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include "mpi/kovalev_k_num_of_orderly_violations/include/header.hpp"

TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_viol_0_int_) {
  const size_t length = 10;
  const int alpha = 1;
  // Create data
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1,0);
  in[1] = -1;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskSeq = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    taskSeq->inputs_count.emplace_back(in.size());
    taskSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskSeq->outputs_count.emplace_back(out.size());
  }
  kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int> tmpTaskSeq(taskSeq);
  ASSERT_EQ(tmpTaskSeq.validation(), true);
  tmpTaskSeq.pre_processing();
  tmpTaskSeq.run();
  tmpTaskSeq.post_processing();
  size_t result = 1;
  ASSERT_EQ(result, out[0]);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  int res = RUN_ALL_TESTS();
  MPI_Finalize();
  return res;
}