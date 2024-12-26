#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>

#include "mpi/shulpin_monte_carlo_integration/include/monte_carlo_integral.hpp"

constexpr double ESTIMATE = 1e-3;

TEST(shulpin_monte_carlo_integration, sin_test) {
  boost::mpi::communicator world;
  double global_integral = 0.0;

  std::shared_ptr<ppc::core::TaskData> task_data_sin = std::make_shared<ppc::core::TaskData>();

  double a = 5.0;
  double b = 11.0;
  int N = 100000;

  if (world.rank() == 0) {
    task_data_sin->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    task_data_sin->inputs_count.emplace_back(1);
    task_data_sin->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    task_data_sin->inputs_count.emplace_back(1);
    task_data_sin->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    task_data_sin->inputs_count.emplace_back(1);
    task_data_sin->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
    task_data_sin->outputs_count.emplace_back(1);
  }

  shulpin_monte_carlo_integration::TestMPITaskParallel parallel_MC_interal(task_data_sin);
  parallel_MC_interal.set_MPI(shulpin_monte_carlo_integration::fsin);
  ASSERT_EQ(parallel_MC_interal.validation(), true);
  parallel_MC_interal.pre_processing();
  parallel_MC_interal.run();
  parallel_MC_interal.post_processing();

  if (world.rank() == 0) {
    double ref_integral = 0.0;

    std::shared_ptr<ppc::core::TaskData> seq_sin_task_data = std::make_shared<ppc::core::TaskData>();
    seq_sin_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    seq_sin_task_data->inputs_count.emplace_back(1);
    seq_sin_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    seq_sin_task_data->inputs_count.emplace_back(1);
    seq_sin_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
    seq_sin_task_data->inputs_count.emplace_back(1);
    seq_sin_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_integral));
    seq_sin_task_data->outputs_count.emplace_back(1);

    shulpin_monte_carlo_integration::TestMPITaskSequential seq_MC_integral(seq_sin_task_data);
    seq_MC_integral.set_seq(shulpin_monte_carlo_integration::fsin);
    ASSERT_EQ(seq_MC_integral.validation(), true);
    seq_MC_integral.pre_processing();
    seq_MC_integral.run();
    seq_MC_integral.post_processing();

    ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
  }
}

TEST(shulpin_monte_carlo_integration, test_cos) {
  boost::mpi::communicator world;
  double global_integral = 0.0;
  std::shared_ptr<ppc::core::TaskData> task_data_cos = std::make_shared<ppc::core::TaskData>();

  double a = -10.0;
  double b = -1.5;
  int n = 100000;

  if (world.rank() == 0) {
    task_data_cos->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    task_data_cos->inputs_count.emplace_back(1);
    task_data_cos->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    task_data_cos->inputs_count.emplace_back(1);
    task_data_cos->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_cos->inputs_count.emplace_back(1);
    task_data_cos->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
    task_data_cos->outputs_count.emplace_back(1);
  }

  shulpin_monte_carlo_integration::TestMPITaskParallel parallel_MC_interal(task_data_cos);
  parallel_MC_interal.set_MPI(shulpin_monte_carlo_integration::fcos);
  ASSERT_EQ(parallel_MC_interal.validation(), true);
  parallel_MC_interal.pre_processing();
  parallel_MC_interal.run();
  parallel_MC_interal.post_processing();

  if (world.rank() == 0) {
    double ref_integral = 0.0;
    std::shared_ptr<ppc::core::TaskData> seq_cos_task_data = std::make_shared<ppc::core::TaskData>();
    seq_cos_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    seq_cos_task_data->inputs_count.emplace_back(1);
    seq_cos_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    seq_cos_task_data->inputs_count.emplace_back(1);
    seq_cos_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    seq_cos_task_data->inputs_count.emplace_back(1);
    seq_cos_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_integral));
    seq_cos_task_data->outputs_count.emplace_back(1);

    shulpin_monte_carlo_integration::TestMPITaskSequential seq_MC_integral(seq_cos_task_data);
    seq_MC_integral.set_seq(shulpin_monte_carlo_integration::fcos);
    ASSERT_EQ(seq_MC_integral.validation(), true);
    seq_MC_integral.pre_processing();
    seq_MC_integral.run();
    seq_MC_integral.post_processing();

    ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
  }
}

TEST(shulpin_monte_carlo_integration, test_two_sin_cos) {
  boost::mpi::communicator world;
  double global_integral = 0.0;
  std::shared_ptr<ppc::core::TaskData> task_data_two_sin_cos = std::make_shared<ppc::core::TaskData>();

  double a = -3.0;
  double b = 4.0;
  int n = 1000000;

  if (world.rank() == 0) {
    task_data_two_sin_cos->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    task_data_two_sin_cos->inputs_count.emplace_back(1);
    task_data_two_sin_cos->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    task_data_two_sin_cos->inputs_count.emplace_back(1);
    task_data_two_sin_cos->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    task_data_two_sin_cos->inputs_count.emplace_back(1);
    task_data_two_sin_cos->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_integral));
    task_data_two_sin_cos->outputs_count.emplace_back(1);
  }

  shulpin_monte_carlo_integration::TestMPITaskParallel parallel_MC_interal(task_data_two_sin_cos);
  parallel_MC_interal.set_MPI(shulpin_monte_carlo_integration::f_two_sin_cos);
  ASSERT_EQ(parallel_MC_interal.validation(), true);
  parallel_MC_interal.pre_processing();
  parallel_MC_interal.run();
  parallel_MC_interal.post_processing();

  if (world.rank() == 0) {
    double ref_integral = 0.0;
    std::shared_ptr<ppc::core::TaskData> seq_two_sin_cos_task_data = std::make_shared<ppc::core::TaskData>();
    seq_two_sin_cos_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    seq_two_sin_cos_task_data->inputs_count.emplace_back(1);
    seq_two_sin_cos_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    seq_two_sin_cos_task_data->inputs_count.emplace_back(1);
    seq_two_sin_cos_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    seq_two_sin_cos_task_data->inputs_count.emplace_back(1);
    seq_two_sin_cos_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_integral));
    seq_two_sin_cos_task_data->outputs_count.emplace_back(1);

    shulpin_monte_carlo_integration::TestMPITaskSequential seq_two_sin_cos(seq_two_sin_cos_task_data);
    seq_two_sin_cos.set_seq(shulpin_monte_carlo_integration::f_two_sin_cos);
    ASSERT_EQ(seq_two_sin_cos.validation(), true);
    seq_two_sin_cos.pre_processing();
    seq_two_sin_cos.run();
    seq_two_sin_cos.post_processing();

    ASSERT_NEAR(ref_integral, global_integral, ESTIMATE);
  }
}

TEST(shulpin_monte_carlo_integration, test_empty_input) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> empty_task_data = std::make_shared<ppc::core::TaskData>();

  shulpin_monte_carlo_integration::TestMPITaskParallel parallel_empty_task(empty_task_data);

  if (world.rank() == 0) {
    ASSERT_FALSE(parallel_empty_task.validation());
  }
}