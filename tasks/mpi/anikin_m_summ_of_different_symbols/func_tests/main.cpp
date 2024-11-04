// Copyright 2024 Anikin Maksim
#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <random>
#include <string>
#include <vector>

#include "mpi/anikin_m_summ_of_different_symbols/include/ops_mpi.hpp"

std::string randomString(size_t size) {
  const std::string characters =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"
      "0123456789"
      "!@#$%^&*()-_=+[]{};:,.<>?";

  std::string result;
  result.reserve(size);

  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution distribution(0, int(characters.size()) - 1);

  for (size_t i = 0; i < size; ++i) {
    result += characters[distribution(generator)];
  }

  return result;
}

TEST(anikin_m_summ_of_different_symbols_mpi, random_same_size) {
  boost::mpi::communicator world;
  std::string input_a;
  std::string input_b;

  auto par_task_output = std::vector<int>(1);
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_a = randomString(1000);
    input_b = randomString(1000);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
    par_task_data->inputs_count.emplace_back(input_a.size());
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
    par_task_data->inputs_count.emplace_back(input_b.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_task_output.data()));
    par_task_data->outputs_count.emplace_back(par_task_output.size());
  }

  auto par_task = anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel(par_task_data);
  ASSERT_TRUE(par_task.validation());
  par_task.pre_processing();
  par_task.run();
  par_task.post_processing();

  if (world.rank() == 0) {
    auto seq_task_output = std::vector<int>(1);
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
    seq_task_data->inputs_count.emplace_back(input_a.size());
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
    seq_task_data->inputs_count.emplace_back(input_b.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_task_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_task_output.size());

    auto seq_task = anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential(seq_task_data);
    ASSERT_TRUE(seq_task.validation());
    seq_task.pre_processing();
    seq_task.run();
    seq_task.post_processing();

    ASSERT_EQ(par_task_output[0], seq_task_output[0]);
  }
}

TEST(anikin_m_summ_of_different_symbols_mpi_func, size_0) {
  boost::mpi::communicator world;
  std::string input_a;
  std::string input_b;

  auto par_task_output = std::vector<int>(1);
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_a = "";
    input_b = "";
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
    par_task_data->inputs_count.emplace_back(input_a.size());
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
    par_task_data->inputs_count.emplace_back(input_b.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_task_output.data()));
    par_task_data->outputs_count.emplace_back(par_task_output.size());
  }

  auto par_task = anikin_m_summ_of_different_symbols_mpi::SumDifSymMPIParallel(par_task_data);
  ASSERT_TRUE(par_task.validation());
  par_task.pre_processing();
  par_task.run();
  par_task.post_processing();

  if (world.rank() == 0) {
    auto seq_task_output = std::vector<int>(1);
    auto seq_task_data = std::make_shared<ppc::core::TaskData>();
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
    seq_task_data->inputs_count.emplace_back(input_a.size());
    seq_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
    seq_task_data->inputs_count.emplace_back(input_b.size());
    seq_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_task_output.data()));
    seq_task_data->outputs_count.emplace_back(seq_task_output.size());

    auto seq_task = anikin_m_summ_of_different_symbols_mpi::SumDifSymMPISequential(seq_task_data);
    ASSERT_TRUE(seq_task.validation());
    seq_task.pre_processing();
    seq_task.run();
    seq_task.post_processing();

    ASSERT_EQ(par_task_output[0], seq_task_output[0]);
  }
}