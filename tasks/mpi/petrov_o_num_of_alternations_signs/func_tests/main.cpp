#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "mpi/petrov_o_num_of_alternations_signs/include/ops_mpi.hpp"

TEST(petrov_o_num_of_alternations_signs_seq, TestAlternations_Simple) {
  std::vector<int> input = {1, -2, 3, -4, 5};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.push_back(output.size());

  petrov_o_num_of_alternations_signs_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output[0], 4);
}

TEST(petrov_o_num_of_alternations_signs_seq, TestAlternations_AllPositive) {
  std::vector<int> input = {1, 2, 3, 4, 5};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.push_back(output.size());

  petrov_o_num_of_alternations_signs_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output[0], 0);
}

TEST(petrov_o_num_of_alternations_signs_seq, TestAlternations_AllNegative) {
  std::vector<int> input = {-1, -2, -3, -4, -5};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.push_back(output.size());

  petrov_o_num_of_alternations_signs_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output[0], 0);
}

TEST(petrov_o_num_of_alternations_signs_seq, TestAlternations_Empty) {
  std::vector<int> input = {};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.push_back(output.size());

  petrov_o_num_of_alternations_signs_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output[0], 0);
}

TEST(petrov_o_num_of_alternations_signs_seq, TestAlternations_OneElement) {
  std::vector<int> input = {1};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.push_back(output.size());

  petrov_o_num_of_alternations_signs_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output[0], 0);
}

TEST(petrov_o_num_of_alternations_signs_seq, TestAlternations_LargeInput) {
  const int size = 1000;
  std::vector<int> input(size);
  std::iota(input.begin(), input.end(), 1);
  for (size_t i = 0; i < input.size(); ++i) {
    if (i % 2 != 0) {
      input[i] *= -1;
    }
  }

  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.push_back(input.size());
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.push_back(output.size());

  petrov_o_num_of_alternations_signs_mpi::SequentialTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  ASSERT_EQ(output[0], static_cast<int>(input.size() - 1));
}

TEST(petrov_o_num_of_alternations_signs_par, TestAlternations_Simple) {
  boost::mpi::communicator world;

  std::vector<int> input = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17, -18, 19, -20};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size());
  }

  petrov_o_num_of_alternations_signs_mpi::ParallelTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  if (world.rank() == 0) {
    ASSERT_EQ(output[0], 19);
  }
}

TEST(petrov_o_num_of_alternations_signs_par, TestAlternations_AllPositive) {
  boost::mpi::communicator world;

  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size());
  }

  petrov_o_num_of_alternations_signs_mpi::ParallelTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  if (world.rank() == 0) {
    ASSERT_EQ(output[0], 0);
  }
}

TEST(petrov_o_num_of_alternations_signs_par, TestAlternations_AllNegative) {
  boost::mpi::communicator world;

  std::vector<int> input = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size());
  }

  petrov_o_num_of_alternations_signs_mpi::ParallelTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  if (world.rank() == 0) {
    ASSERT_EQ(output[0], 0);
  }
}

TEST(petrov_o_num_of_alternations_signs_par, TestAlternations_Empty) {
  boost::mpi::communicator world;

  std::vector<int> input = {};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size());
  }

  petrov_o_num_of_alternations_signs_mpi::ParallelTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  if (world.rank() == 0) {
    ASSERT_EQ(output[0], 0);
  }
}

TEST(petrov_o_num_of_alternations_signs_par, TestAlternations_OneElement) {
  boost::mpi::communicator world;

  std::vector<int> input = {1};
  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size());
  }

  petrov_o_num_of_alternations_signs_mpi::ParallelTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  if (world.rank() == 0) {
    ASSERT_EQ(output[0], 0);
  }
}

TEST(petrov_o_num_of_alternations_signs_par, TestAlternations_LargeInput) {
  boost::mpi::communicator world;

  const int size = 1000;
  std::vector<int> input(size);
  std::iota(input.begin(), input.end(), 1);
  for (size_t i = 0; i < input.size(); ++i) {
    if (i % 2 != 0) {
      input[i] *= -1;
    }
  }

  std::vector<int> output(1);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    taskData->inputs_count.push_back(input.size());
    taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output.data()));
    taskData->outputs_count.push_back(output.size());
  }

  petrov_o_num_of_alternations_signs_mpi::ParallelTask task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
  if (world.rank() == 0) {
    ASSERT_EQ(output[0], static_cast<int>(input.size() - 1));
  }
}

TEST(petrov_o_num_of_alternations_signs_par, TestAlternations_Random) {
  boost::mpi::communicator world;

  const int size = 1000;
  std::vector<int> input(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-100, 100);

  for (int i = 0; i < size; ++i) {
    input[i] = dist(gen);
  }

  std::vector<int> seq_output(1);
  std::vector<int> par_output(1);

  // Sequential run
  std::shared_ptr<ppc::core::TaskData> seq_taskData = std::make_shared<ppc::core::TaskData>();
  seq_taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
  seq_taskData->inputs_count.push_back(input.size());
  seq_taskData->outputs.push_back(reinterpret_cast<uint8_t*>(seq_output.data()));
  seq_taskData->outputs_count.push_back(seq_output.size());

  petrov_o_num_of_alternations_signs_mpi::SequentialTask seq_task(seq_taskData);
  ASSERT_TRUE(seq_task.validation());
  ASSERT_TRUE(seq_task.pre_processing());
  ASSERT_TRUE(seq_task.run());
  ASSERT_TRUE(seq_task.post_processing());

  // Parallel run
  std::shared_ptr<ppc::core::TaskData> par_taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    par_taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input.data()));
    par_taskData->inputs_count.push_back(input.size());
    par_taskData->outputs.push_back(reinterpret_cast<uint8_t*>(par_output.data()));
    par_taskData->outputs_count.push_back(par_output.size());
  }

  petrov_o_num_of_alternations_signs_mpi::ParallelTask par_task(par_taskData);
  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    ASSERT_EQ(par_output[0], seq_output[0]);
  }
}
