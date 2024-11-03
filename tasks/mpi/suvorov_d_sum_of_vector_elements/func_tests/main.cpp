// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/suvorov_d_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Normal_Vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    int count_size_vector = 120;
    // The number of processes should be less than the number of elements
    if (world.size() >= count_size_vector) {
      count_size_vector = 2 * world.size();
    }
    global_vec = suvorov_d_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);

    // Calculating the sum sequentially for verification
    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);

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

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Empty_Vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Calculating the sum sequentially for verification
    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);

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

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Single_Elementr) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    const int count_size_vector = 1;
    global_vec = suvorov_d_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);

    // Calculating the sum sequentially for verification
    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);

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

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_When_Process_Count_More_Than_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // The number of processes must be greater than the number of elements
    const int count_size_vector = world.size() / 2;
    global_vec = suvorov_d_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);

    // Calculating the sum sequentially for verification
    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);

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

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_When_Process_Count_Equal_To_Elements) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // The number of processes must be equal to the number of elements
    const int count_size_vector = world.size();
    global_vec = suvorov_d_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);

    // Calculating the sum sequentially for verification
    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);

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

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Zero_Vector) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // Creating a zero vector
    const int count_size_vector = 120;
    global_vec = std::vector(count_size_vector, 0);

    // Calculating the sum sequentially for verification
    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);

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

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Multiple_Of_Num_Proc_And_Num_Elems) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // The number of elements must be a multiple of the number of processes
    const int count_size_vector = 3 * world.size();
    global_vec = suvorov_d_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);

    // Calculating the sum sequentially for verification
    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);

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

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
}

TEST(suvorov_d_sum_of_vector_elements_mpi, Test_Sum_With_Not_Multiple_Of_Num_Proc_And_Num_Elems) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_sum(1, 0);
  int right_result = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    // The number of elements should not be a multiple of the number of processes
    int count_size_vector = 120;
    if (world.size() < count_size_vector) {
      if (count_size_vector % world.size() == 0) {
        std::random_device dev;
        std::mt19937 gen(dev());
        std::uniform_int_distribution<int> dist(1, world.size() - 1);

        count_size_vector -= dist(gen);
      }
    }
    global_vec = suvorov_d_sum_of_vector_elements_mpi::getRandomVector(count_size_vector);

    // Calculating the sum sequentially for verification
    right_result = std::accumulate(global_vec.begin(), global_vec.end(), 0);

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

  if (world.rank() == 0) {
    ASSERT_EQ(right_result, global_sum[0]);
  }
}