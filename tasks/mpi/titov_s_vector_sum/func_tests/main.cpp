// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/titov_s_vector_sum/include/ops_mpi.hpp"

TEST(titov_s_vector_sum_mpi, Test_Sum_100) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 100;
    global_vec = titov_s_vector_sum_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  titov_s_vector_sum_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_EQ(MPIVectorSumParallel.validation(), true);
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_sum(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    titov_s_vector_sum_mpi::MPIVectorSumSequential MPIVectorSumSequential(taskDataSeq);
    ASSERT_EQ(MPIVectorSumSequential.validation(), true);
    MPIVectorSumSequential.pre_processing();
    MPIVectorSumSequential.run();
    MPIVectorSumSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(titov_s_vector_sum_mpi, Test_Sum_1000) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 1000;
    global_vec = titov_s_vector_sum_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  titov_s_vector_sum_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_EQ(MPIVectorSumParallel.validation(), true);
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_sum(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    titov_s_vector_sum_mpi::MPIVectorSumSequential MPIVectorSumSequential(taskDataSeq);
    ASSERT_EQ(MPIVectorSumSequential.validation(), true);
    MPIVectorSumSequential.pre_processing();
    MPIVectorSumSequential.run();
    MPIVectorSumSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(titov_s_vector_sum_mpi, Test_Sum_100000) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 100000;
    global_vec = titov_s_vector_sum_mpi::getRandomVector(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  titov_s_vector_sum_mpi::MPIVectorSumParallel MPIVectorSumParallel(taskDataPar);
  ASSERT_EQ(MPIVectorSumParallel.validation(), true);
  MPIVectorSumParallel.pre_processing();
  MPIVectorSumParallel.run();
  MPIVectorSumParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_sum(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    titov_s_vector_sum_mpi::MPIVectorSumSequential MPIVectorSumSequential(taskDataSeq);
    ASSERT_EQ(MPIVectorSumSequential.validation(), true);
    MPIVectorSumSequential.pre_processing();
    MPIVectorSumSequential.run();
    MPIVectorSumSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}
