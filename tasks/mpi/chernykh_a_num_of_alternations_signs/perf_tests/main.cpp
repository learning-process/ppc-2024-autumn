#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/chernykh_a_num_of_alternations_signs/include/ops_mpi.hpp"

TEST(chernykh_a_num_of_alternations_signs_mpi, test_pipeline_run) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>(1, 0);

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = std::vector<int>(30'000'000, 0);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = std::make_shared<chernykh_a_num_of_alternations_signs_mpi::ParallelTask>(par_task_data);

  ASSERT_TRUE(par_task->validation());
  ASSERT_TRUE(par_task->pre_processing());
  ASSERT_TRUE(par_task->run());
  ASSERT_TRUE(par_task->post_processing());

  // Create PerfAttributes
  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = boost::mpi::timer();
  perf_attributes->current_timer = [&] { return start.elapsed(); };

  // Create PerfResults
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(par_task);

  perf_analyzer->pipeline_run(perf_attributes, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    ASSERT_EQ(0, par_output[0]);
  }
}

TEST(chernykh_a_num_of_alternations_signs_mpi, test_task_run) {
  auto world = boost::mpi::communicator();

  // Create data
  auto input = std::vector<int>();
  auto par_output = std::vector<int>(1, 0);

  // Create TaskData
  auto par_task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input = std::vector<int>(30'000'000, 0);
    par_task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    par_task_data->inputs_count.emplace_back(input.size());
    par_task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(par_output.data()));
    par_task_data->outputs_count.emplace_back(par_output.size());
  }

  // Create Task
  auto par_task = std::make_shared<chernykh_a_num_of_alternations_signs_mpi::ParallelTask>(par_task_data);

  ASSERT_TRUE(par_task->validation());
  ASSERT_TRUE(par_task->pre_processing());
  ASSERT_TRUE(par_task->run());
  ASSERT_TRUE(par_task->post_processing());

  // Create PerfAttributes
  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = boost::mpi::timer();
  perf_attributes->current_timer = [&] { return start.elapsed(); };

  // Create PerfResults
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(par_task);

  perf_analyzer->task_run(perf_attributes, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    ASSERT_EQ(0, par_output[0]);
  }
}
