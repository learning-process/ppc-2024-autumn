#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/bessonov_e_star_topology/include/ops_mpi.hpp"

TEST(mpi_bessonov_e_star_topology_perf_test, test_pipeline_run) {
  const int data_size = 10000000;
  boost::mpi::communicator world;
  std::vector<int> input_data(data_size);
  std::vector<int> output_data(data_size);
  std::vector<int> traversal_order(data_size);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine random_engine(random_device());
    std::uniform_int_distribution<int> distribution(0, input_data.size());
    std::generate(input_data.begin(), input_data.end(),
                  [&distribution, &random_engine] { return distribution(random_engine); });
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data->outputs_count.emplace_back(output_data.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(traversal_order.data()));
    task_data->outputs_count.emplace_back(traversal_order.size());
  }
  auto mpi_task_parallel = std::make_shared<bessonov_e_star_topology_mpi::TestMPITaskParallel>(task_data);

  // Create Perf attributes
  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  const boost::mpi::timer timer;
  perf_attributes->current_timer = [&] { return timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(mpi_task_parallel);
  perf_analyzer->pipeline_run(perf_attributes, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    ASSERT_EQ(true, std::equal(input_data.begin(), input_data.end(), output_data.begin(), output_data.end()));
  }
}

TEST(mpi_bessonov_e_star_topology_perf_test, test_task_run) {
  const int data_size = 10000000;
  boost::mpi::communicator world;
  std::vector<int> input_data(data_size);
  std::vector<int> output_data(data_size);
  std::vector<int> traversal_order(data_size);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine random_engine(random_device());
    std::uniform_int_distribution<int> distribution(0, input_data.size());
    std::generate(input_data.begin(), input_data.end(),
                  [&distribution, &random_engine] { return distribution(random_engine); });
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.emplace_back(input_data.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data->outputs_count.emplace_back(output_data.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(traversal_order.data()));
    task_data->outputs_count.emplace_back(traversal_order.size());
  }
  auto mpi_task_parallel = std::make_shared<bessonov_e_star_topology_mpi::TestMPITaskParallel>(task_data);

  // Create Perf attributes
  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  const boost::mpi::timer timer;
  perf_attributes->current_timer = [&] { return timer.elapsed(); };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(mpi_task_parallel);
  perf_analyzer->task_run(perf_attributes, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    ASSERT_EQ(true, std::equal(input_data.begin(), input_data.end(), output_data.begin(), output_data.end()));
  }
}
