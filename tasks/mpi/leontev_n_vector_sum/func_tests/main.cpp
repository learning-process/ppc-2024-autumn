// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/leontev_n_vector_sum/include/ops_mpi.hpp"

namespace leontev_n_vec_sum_mpi {
std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}
}  // namespace leontev_n_vec_sum_mpi

inline void taskEmplacement(std::shared_ptr<ppc::core::TaskData>& taskDataPar, std::vector<int>& global_vec,
                            std::vector<int32_t>& global_sum) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
  taskDataPar->outputs_count.emplace_back(global_sum.size());
}

TEST(leontev_n_vec_sum_mpi, sum_mpi_50elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 50;
    global_vec = leontev_n_vec_sum_mpi::getRandomVector(vector_size);
    taskEmplacement(taskDataPar, global_vec, global_sum);
  }
  leontev_n_vec_sum_mpi::MPIVecSumParallel MPIVecSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVecSumParallel.validation());
  MPIVecSumParallel.pre_processing();
  MPIVecSumParallel.run();
  MPIVecSumParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_sum(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, global_vec, reference_sum);
    // Create Task
    leontev_n_vec_sum_mpi::MPIVecSumSequential MPIVecSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVecSumSequential.validation());
    MPIVecSumSequential.pre_processing();
    MPIVecSumSequential.run();
    MPIVecSumSequential.post_processing();
    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}
TEST(leontev_n_vec_sum_mpi, sum_mpi_0elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, global_vec, global_sum);
  }
  leontev_n_vec_sum_mpi::MPIVecSumParallel MPIVecSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVecSumParallel.validation());
  MPIVecSumParallel.pre_processing();
  MPIVecSumParallel.run();
  MPIVecSumParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(global_sum[0], 0);
  }
}
TEST(leontev_n_vec_sum_mpi, sum_mpi_1000elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 1000;
    global_vec = leontev_n_vec_sum_mpi::getRandomVector(vector_size);
    taskEmplacement(taskDataPar, global_vec, global_sum);
  }
  leontev_n_vec_sum_mpi::MPIVecSumParallel MPIVecSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVecSumParallel.validation());
  MPIVecSumParallel.pre_processing();
  MPIVecSumParallel.run();
  MPIVecSumParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_sum(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, global_vec, reference_sum);
    // Create Task
    leontev_n_vec_sum_mpi::MPIVecSumSequential MPIVecSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVecSumSequential.validation());
    MPIVecSumSequential.pre_processing();
    MPIVecSumSequential.run();
    MPIVecSumSequential.post_processing();
    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}
TEST(leontev_n_vec_sum_mpi, sum_mpi_20000elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 20000;
    global_vec = leontev_n_vec_sum_mpi::getRandomVector(vector_size);
    taskEmplacement(taskDataPar, global_vec, global_sum);
  }
  leontev_n_vec_sum_mpi::MPIVecSumParallel MPIVecSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVecSumParallel.validation());
  MPIVecSumParallel.pre_processing();
  MPIVecSumParallel.run();
  MPIVecSumParallel.post_processing();
  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_sum(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, global_vec, reference_sum);
    // Create Task
    leontev_n_vec_sum_mpi::MPIVecSumSequential MPIVecSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVecSumSequential.validation());
    MPIVecSumSequential.pre_processing();
    MPIVecSumSequential.run();
    MPIVecSumSequential.post_processing();
    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}
TEST(leontev_n_vec_sum_mpi, sum_mpi_1elem) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int vector_size = 1;
    global_vec = leontev_n_vec_sum_mpi::getRandomVector(vector_size);
    taskEmplacement(taskDataPar, global_vec, global_sum);
  }
  leontev_n_vec_sum_mpi::MPIVecSumParallel MPIVecSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVecSumParallel.validation());
  MPIVecSumParallel.pre_processing();
  MPIVecSumParallel.run();
  MPIVecSumParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference_sum(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, global_vec, reference_sum);
    leontev_n_vec_sum_mpi::MPIVecSumSequential MPIVecSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVecSumSequential.validation());
    MPIVecSumSequential.pre_processing();
    MPIVecSumSequential.run();
    MPIVecSumSequential.post_processing();
    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}
TEST(leontev_n_vec_sum_mpi, sum_mpi_0elem_seq_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1, 0);
  std::vector<int32_t> global_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskEmplacement(taskDataPar, global_vec, global_sum);
  }
  leontev_n_vec_sum_mpi::MPIVecSumParallel MPIVecSumParallel(taskDataPar);
  ASSERT_TRUE(MPIVecSumParallel.validation());
  MPIVecSumParallel.pre_processing();
  MPIVecSumParallel.run();
  MPIVecSumParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference_sum(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskEmplacement(taskDataSeq, global_vec, reference_sum);
    leontev_n_vec_sum_mpi::MPIVecSumSequential MPIVecSumSequential(taskDataSeq);
    ASSERT_TRUE(MPIVecSumSequential.validation());
    MPIVecSumSequential.pre_processing();
    MPIVecSumSequential.run();
    MPIVecSumSequential.post_processing();
    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}
