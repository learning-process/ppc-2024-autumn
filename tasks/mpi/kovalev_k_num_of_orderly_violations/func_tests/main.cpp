// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include "mpi/kovalev_k_num_of_orderly_violations/include/header.hpp"


TEST(kovalev_k_num_of_orderly_violations_mpi, Test_NoOV_viol_0_int_) {
  const size_t length = 10;
  const int alpha = 1;

  // Create data
  std::vector<int> in(length, alpha);
  std::vector<size_t> out(1);
  in[1]=-1;
  
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
	
  ASSERT_EQ(1, out[0]);
}

int main(int argc, char **argv) {
	
  MPI_Init(&argc, &argv);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  
  if(rank!=0){
    delete listeners.Release(listeners.default_result_printer());
  }
  
  int res = RUN_ALL_TESTS();
  
  MPI_Finalize();

  return res;
}





/*TEST(baranov_a_num_of_orderly_violations_mpi, Test_viol_0_int) {
  const int N = 0;
  // Create data
  boost::mpi::communicator world;
  std::vector<int> arr(N);
  std::vector<int> out(1);
  std::shared_ptr<ppc::core::TaskData> data_seq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device rd;
    std::default_random_engine reng(rd());
    std::uniform_int_distribution<int> dist(0, arr.size());
    std::generate(arr.begin(), arr.end(), [&dist, &reng] { return dist(reng); });
    data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    data_seq->inputs_count.emplace_back(arr.size());
    data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    data_seq->outputs_count.emplace_back(1);
  }
  baranov_a_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int> test1(data_seq);
  ASSERT_EQ(test1.validation(), true);
  test1.pre_processing();
  test1.run();
  test1.post_processing();
  int num = test1.seq_proc(arr);
  ASSERT_EQ(out[0], num);
}*/