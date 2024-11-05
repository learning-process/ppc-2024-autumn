// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/suvorov_d_sum_of_vector_elements/include/ops_mpi.hpp"

std::vector<int> get_random_vector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-1000, 1000);

  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }

  return vec;
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Normal_Vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    int count_size_vector = 120;
    // The number of processes should be less than the number of elements
    if (world.size() >= count_size_vector) {
      count_size_vector = 2 * world.size();
    }
    global_vec = get_random_vector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  // Execution of addition
  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  // Calculating the sum sequentially for verification
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
    suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
    ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
    SumOfVectorElementsSeq.pre_processing();
    SumOfVectorElementsSeq.run();
    SumOfVectorElementsSeq.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Empty_Vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  // Execution of addition
  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  // Calculating the sum sequentially for verification
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
    suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
    ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
    SumOfVectorElementsSeq.pre_processing();
    SumOfVectorElementsSeq.run();
    SumOfVectorElementsSeq.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Single_Elementr) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 1;
    global_vec = get_random_vector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  // Execution of addition
  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  // Calculating the sum sequentially for verification
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
    suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
    ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
    SumOfVectorElementsSeq.pre_processing();
    SumOfVectorElementsSeq.run();
    SumOfVectorElementsSeq.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_When_Process_Count_More_Than_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // The number of processes must be greater than the number of elements
    const int count_size_vector = world.size() / 2;
    global_vec = get_random_vector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  // Execution of addition
  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  // Calculating the sum sequentially for verification
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
    suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
    ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
    SumOfVectorElementsSeq.pre_processing();
    SumOfVectorElementsSeq.run();
    SumOfVectorElementsSeq.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_When_Process_Count_Equal_To_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // The number of processes must be equal to the number of elements
    const int count_size_vector = world.size();
    global_vec = get_random_vector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  // Execution of addition
  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  // Calculating the sum sequentially for verification
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
    suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
    ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
    SumOfVectorElementsSeq.pre_processing();
    SumOfVectorElementsSeq.run();
    SumOfVectorElementsSeq.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Zero_Vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Creating a zero vector
    const int count_size_vector = 120;
    global_vec = std::vector(count_size_vector, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  // Execution of addition
  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  // Calculating the sum sequentially for verification
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
    suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
    ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
    SumOfVectorElementsSeq.pre_processing();
    SumOfVectorElementsSeq.run();
    SumOfVectorElementsSeq.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Multiple_Of_Num_Proc_And_Num_Elems) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // The number of elements must be a multiple of the number of processes
    const int count_size_vector = 3 * world.size();
    global_vec = get_random_vector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  // Execution of addition
  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  // Calculating the sum sequentially for verification
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
    suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
    ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
    SumOfVectorElementsSeq.pre_processing();
    SumOfVectorElementsSeq.run();
    SumOfVectorElementsSeq.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Not_Multiple_Of_Num_Proc_And_Num_Elems) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // The number of elements should not be a multiple of the number of processes
    // Set prime number
    int count_size_vector = 101;

    global_vec = get_random_vector(count_size_vector);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  // Execution of addition
  suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_parallel SumOfVectorElementsParallel(taskDataPar);
  ASSERT_EQ(SumOfVectorElementsParallel.validation(), true);
  SumOfVectorElementsParallel.pre_processing();
  SumOfVectorElementsParallel.run();
  SumOfVectorElementsParallel.post_processing();

  // Calculating the sum sequentially for verification
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
    suvorov_d_sum_of_vector_elements_mpi::Sum_of_vector_elements_seq SumOfVectorElementsSeq(taskDataSeq);
    ASSERT_EQ(SumOfVectorElementsSeq.validation(), true);
    SumOfVectorElementsSeq.pre_processing();
    SumOfVectorElementsSeq.run();
    SumOfVectorElementsSeq.post_processing();

    ASSERT_EQ(reference_sum[0], global_sum[0]);
  }
}