#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <random>
#include <vector>

#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

TEST(petrov_a_Shell_sort_mpi, test_task_run_mpi) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const size_t size = 10000;
  std::vector<int> input_data;
  std::vector<int> output_data(size);

  if (rank == 0) {
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    input_data.resize(size);
    for (size_t i = 0; i < size; ++i) {
      input_data[i] = dist(rng);
    }
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(input_data.data());
  taskData->inputs_count.push_back(size);
  taskData->outputs.push_back(output_data.data());
  taskData->outputs_count.push_back(size);

  petrov_a_Shell_sort_mpi::TestTaskMPI task(taskData);

  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (rank == 0) {
    std::vector<int> expected_data = input_data;
    std::sort(expected_data.begin(), expected_data.end());
    EXPECT_EQ(output_data, expected_data);
  }
}

TEST(petrov_a_Shell_sort_mpi, test_pipeline_run_mpi) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_elements = 10000;
  const int elements_per_proc = total_elements / size;

  std::vector<int> global_data;
  std::vector<int> local_data(elements_per_proc);

  if (rank == 0) {
    global_data.resize(total_elements);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (int i = 0; i < total_elements; ++i) {
      global_data[i] = dist(rng);
    }
  }

  MPI_Scatter(global_data.data(), elements_per_proc, MPI_INT, local_data.data(), elements_per_proc, MPI_INT, 0,
              MPI_COMM_WORLD);

  petrov_a_Shell_sort_mpi::TestTaskMPI task(nullptr);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(local_data.data());
  taskData->inputs_count.push_back(local_data.size());
  taskData->outputs.push_back(local_data.data());
  taskData->outputs_count.push_back(local_data.size());

  task = petrov_a_Shell_sort_mpi::TestTaskMPI(taskData);

  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  if (rank == 0) {
    global_data.resize(total_elements);
  }

  MPI_Gather(local_data.data(), elements_per_proc, MPI_INT, global_data.data(), elements_per_proc, MPI_INT, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    EXPECT_TRUE(std::is_sorted(global_data.begin(), global_data.end()));
  }
}
