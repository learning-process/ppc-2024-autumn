#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/lavrentyev_a_line_topology/include/ops_mpi.hpp"

namespace lavrentyev_a_line_topology_mpi {

std::vector<int> generate_random_data(int count, int lower_bound = -1000, int upper_bound = 1000) {
  std::vector<int> data(count);
  std::mt19937 random_engine(std::random_device{}());
  std::uniform_int_distribution<int> distribution(lower_bound, upper_bound);
  for (int& value : data) {
    value = distribution(random_engine);
  }
  return data;
}

TEST(lavrentyev_a_line_topology_mpi, task_run) {
  boost::mpi::communicator mpi_comm;

  if (mpi_comm.size() < 2) return;

  int total_elements = 1000000;
  int start_rank = 0;
  int end_rank = mpi_comm.size() - 1;

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs_count.push_back(start_rank);
  data->inputs_count.push_back(end_rank);
  data->inputs_count.push_back(total_elements);

  std::vector<int> input_values;
  std::vector<int> result_values(total_elements);
  std::vector<int> processing_order;

  if (mpi_comm.rank() == start_rank) {
    input_values = lavrentyev_a_line_topology_mpi::generate_random_data(total_elements);
    data->inputs.push_back(reinterpret_cast<uint8_t*>(input_values.data()));
  }

  if (mpi_comm.rank() == end_rank) {
    processing_order.resize(end_rank - start_rank + 1);
    data->outputs = {reinterpret_cast<uint8_t*>(result_values.data()),
                     reinterpret_cast<uint8_t*>(processing_order.data())};
    data->outputs_count.push_back(result_values.size());
    data->outputs_count.push_back(processing_order.size());
  }

  auto task = std::make_shared<lavrentyev_a_line_topology_mpi::TestMPITaskParallel>(data);

  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = 10;
  boost::mpi::timer timer;
  performance_attributes->current_timer = [&] { return timer.elapsed(); };

  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  auto performance_analyzer = std::make_shared<ppc::core::Perf>(task);
  performance_analyzer->task_run(performance_attributes, performance_results);

  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  if (mpi_comm.rank() == end_rank) {
    ppc::core::Perf::print_perf_statistic(performance_results);

    ASSERT_EQ(input_values, result_values);
    for (size_t i = 0; i < processing_order.size(); ++i) {
      ASSERT_EQ(processing_order[i], start_rank + static_cast<int>(i));
    }
  }
}

TEST(lavrentyev_a_line_topology_mpi, pipeline_run) {
  boost::mpi::communicator mpi_comm;

  if (mpi_comm.size() < 2) return;

  int total_elements = 10'000'000;
  int start_rank = 0;
  int end_rank = mpi_comm.size() - 1;

  auto data = std::make_shared<ppc::core::TaskData>();
  data->inputs_count.push_back(start_rank);
  data->inputs_count.push_back(end_rank);
  data->inputs_count.push_back(total_elements);

  std::vector<int> input_values;
  std::vector<int> result_values(total_elements);
  std::vector<int> trace_path;

  if (mpi_comm.rank() == start_rank) {
    input_values = lavrentyev_a_line_topology_mpi::generate_random_data(total_elements);
    data->inputs.push_back(reinterpret_cast<uint8_t*>(input_values.data()));
  }

  if (mpi_comm.rank() == end_rank) {
    trace_path.resize(end_rank - start_rank + 1);
    data->outputs = {reinterpret_cast<uint8_t*>(result_values.data()),
                     reinterpret_cast<uint8_t*>(trace_path.data())};
    data->outputs_count.push_back(result_values.size());
    data->outputs_count.push_back(trace_path.size());
  }

  auto parallel_task = std::make_shared<lavrentyev_a_line_topology_mpi::TestMPITaskParallel>(data);

  auto perf_attrs = std::make_shared<ppc::core::PerfAttr>();
  perf_attrs->num_running = 5;
  boost::mpi::timer task_time;
  perf_attrs->current_timer = [&] { return task_time.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(parallel_task);

  perf_analyzer->pipeline_run(perf_attrs, perf_results);

  if (mpi_comm.rank() == end_rank) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    ASSERT_EQ(input_values, result_values);
    for (size_t i = 0; i < trace_path.size(); ++i) {
      ASSERT_EQ(trace_path[i], start_rank + static_cast<int>(i));
    }
  }
}

}  // namespace lavrentyev_a_line_topology_mpi
