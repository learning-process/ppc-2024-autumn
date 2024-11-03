// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/suvorov_d_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 10;
    global_vec = suvorov_d_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);
    for (int element : global_vec) { // NEED_TO_DELETE
        std::cout << element << " "; // NEED_TO_DELETE
    } // NEED_TO_DELETE
    std::cout << std::endl; // NEED_TO_DELETE

    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);
    
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  for (int element : global_vec) { // NEED_TO_DELETE
    std::cout << element << " "; // NEED_TO_DELETE
  } // NEED_TO_DELETE
  std::cout << std::endl; // NEED_TO_DELETE

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
  // if (world.rank() == 0) {
  //   // Create data
  //   std::vector<int32_t> reference_sum(1, 0);

  //   // Create TaskData
  //   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  //   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  //   taskDataSeq->inputs_count.emplace_back(global_vec.size());
  //   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
  //   taskDataSeq->outputs_count.emplace_back(reference_sum.size());

  //   // Create Task
  //   suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
  //   ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
  //   SumOfVectorElementsSeq.pre_processing();
  //   std::cout << "RUN START\n" << std::endl << std::endl;
  //   SumOfVectorElementsSeq.run();
  //   SumOfVectorElementsSeq.post_processing();

  //   ASSERT_EQ(reference_sum[0], global_sum[0]);
  // }
}
