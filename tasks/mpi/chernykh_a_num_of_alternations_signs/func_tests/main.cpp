#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <vector>

#include "mpi/chernykh_a_num_of_alternations_signs/include/ops_mpi.hpp"

TEST(chernykh_a_num_of_alternations_signs_mpi, random_input) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>(1, 0);

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = chernykh_a_num_of_alternations_signs_mpi::getRandomVector(100'000);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = chernykh_a_num_of_alternations_signs_mpi::ParallelTask(par_task_data);

  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    // Create data
    auto seq_output = std::vector<int>(1, 0);

    // Create TaskData
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    // Create Task
    auto seq_task = chernykh_a_num_of_alternations_signs_mpi::SequentialTask(seq_task_data);

    ASSERT_TRUE(seq_task.validation());
    ASSERT_TRUE(seq_task.pre_processing());
    ASSERT_TRUE(seq_task.run());
    ASSERT_TRUE(seq_task.post_processing());
    ASSERT_EQ(seq_output[0], par_output[0]);
  }
}

TEST(chernykh_a_num_of_alternations_signs_mpi, large_random_input) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>(1, 0);

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = chernykh_a_num_of_alternations_signs_mpi::getRandomVector(1'000'000);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = chernykh_a_num_of_alternations_signs_mpi::ParallelTask(par_task_data);

  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    // Create data
    auto seq_output = std::vector<int>(1, 0);

    // Create TaskData
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    // Create Task
    auto seq_task = chernykh_a_num_of_alternations_signs_mpi::SequentialTask(seq_task_data);

    ASSERT_TRUE(seq_task.validation());
    ASSERT_TRUE(seq_task.pre_processing());
    ASSERT_TRUE(seq_task.run());
    ASSERT_TRUE(seq_task.post_processing());
    ASSERT_EQ(seq_output[0], par_output[0]);
  }
}

TEST(chernykh_a_num_of_alternations_signs_mpi, input_size_less_than_two_fails_validation) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>(1, 0);

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = chernykh_a_num_of_alternations_signs_mpi::ParallelTask(par_task_data);

  if (world.rank() == 0) {
    ASSERT_FALSE(par_task.validation());
  }

  if (world.rank() == 0) {
    // Create data
    auto seq_output = std::vector<int>(1, 0);

    // Create TaskData
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    // Create Task
    auto seq_task = chernykh_a_num_of_alternations_signs_mpi::SequentialTask(seq_task_data);

    ASSERT_FALSE(seq_task.validation());
  }
}

TEST(chernykh_a_num_of_alternations_signs_mpi, output_size_not_equal_one_fails_validation) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>();

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = chernykh_a_num_of_alternations_signs_mpi::getRandomVector(1000);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = chernykh_a_num_of_alternations_signs_mpi::ParallelTask(par_task_data);

  if (world.rank() == 0) {
    ASSERT_FALSE(par_task.validation());
  }

  if (world.rank() == 0) {
    // Create data
    auto seq_output = std::vector<int>();

    // Create TaskData
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    // Create Task
    auto seq_task = chernykh_a_num_of_alternations_signs_mpi::SequentialTask(seq_task_data);

    ASSERT_FALSE(seq_task.validation());
  }
}

TEST(chernykh_a_num_of_alternations_signs_mpi, all_elements_are_equal) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>(1, 0);

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = std::vector<int>(1000, 0);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = chernykh_a_num_of_alternations_signs_mpi::ParallelTask(par_task_data);

  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    // Create data
    auto seq_output = std::vector<int>(1, 0);

    // Create TaskData
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    // Create Task
    auto seq_task = chernykh_a_num_of_alternations_signs_mpi::SequentialTask(seq_task_data);

    ASSERT_TRUE(seq_task.validation());
    ASSERT_TRUE(seq_task.pre_processing());
    ASSERT_TRUE(seq_task.run());
    ASSERT_TRUE(seq_task.post_processing());
    ASSERT_EQ(seq_output[0], par_output[0]);
  }
}

TEST(chernykh_a_num_of_alternations_signs_mpi, sign_change_at_borders_of_two_chunks) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>(1, 0);

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = std::vector<int>{1, 1, 1, 1, 1, -1, -1, -1, -1, -1};
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = chernykh_a_num_of_alternations_signs_mpi::ParallelTask(par_task_data);

  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    // Create data
    auto seq_output = std::vector<int>(1, 0);

    // Create TaskData
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    // Create Task
    auto seq_task = chernykh_a_num_of_alternations_signs_mpi::SequentialTask(seq_task_data);

    ASSERT_TRUE(seq_task.validation());
    ASSERT_TRUE(seq_task.pre_processing());
    ASSERT_TRUE(seq_task.run());
    ASSERT_TRUE(seq_task.post_processing());
    ASSERT_EQ(seq_output[0], par_output[0]);
  }
}

TEST(chernykh_a_num_of_alternations_signs_mpi, sign_change_at_borders_of_three_chunks) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>(1, 0);

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = std::vector<int>{1, 1, 1, -1, -1, -1, 1, 1, 1};
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = chernykh_a_num_of_alternations_signs_mpi::ParallelTask(par_task_data);

  ASSERT_TRUE(par_task.validation());
  ASSERT_TRUE(par_task.pre_processing());
  ASSERT_TRUE(par_task.run());
  ASSERT_TRUE(par_task.post_processing());

  if (world.rank() == 0) {
    // Create data
    auto seq_output = std::vector<int>(1, 0);

    // Create TaskData
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    seq_task_data->inputs_count.emplace_back(input.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_output.size());

    // Create Task
    auto seq_task = chernykh_a_num_of_alternations_signs_mpi::SequentialTask(seq_task_data);

    ASSERT_TRUE(seq_task.validation());
    ASSERT_TRUE(seq_task.pre_processing());
    ASSERT_TRUE(seq_task.run());
    ASSERT_TRUE(seq_task.post_processing());
    ASSERT_EQ(seq_output[0], par_output[0]);
  }
}