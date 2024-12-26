#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/yasakova_t_quick_sort_with_simple_merge/include/ops_mpi.hpp"

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, pipeline_run) {
  boost::mpi::communicator mpi_comm;
  int vector_size = 10000;
  std::vector<int> input_data(vector_size);
  std::vector<int> output_data;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size));
  task_data->inputs_count.emplace_back(1);

  if (mpi_comm.rank() == 0) {
    int current_value = vector_size;
    std::generate(input_data.begin(), input_data.end(), [&current_value]() { return current_value--; });

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.emplace_back(input_data.size());

    output_data.resize(vector_size);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data->outputs_count.emplace_back(output_data.size());
  }

  auto parallel_sort_task =
      std::make_shared<yasakova_t_quick_sort_with_simple_merge_mpi::SimpleMergeQuicksort>(task_data);

  ASSERT_TRUE(parallel_sort_task->validation());
  parallel_sort_task->pre_processing();
  parallel_sort_task->run();
  parallel_sort_task->post_processing();

  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = 10;
  const boost::mpi::timer timer;
  performance_attributes->current_timer = [&] { return timer.elapsed(); };
  auto performance_results = std::make_shared<ppc::core::PerfResults>();
  auto performance_analyzer = std::make_shared<ppc::core::Perf>(parallel_sort_task);
  performance_analyzer->pipeline_run(performance_attributes, performance_results);

  if (mpi_comm.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(performance_results);
    std::sort(input_data.begin(), input_data.end());
    EXPECT_EQ(input_data, output_data);
  }
}

TEST(yasakova_t_quick_sort_with_simple_merge_mpi, test_task_execution) {
  boost::mpi::communicator mpi_comm;
  int vector_size = 10000;
  std::vector<int> input_data(vector_size);
  std::vector<int> output_data;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size));
  task_data->inputs_count.emplace_back(1);

  if (mpi_comm.rank() == 0) {
    int current_value = vector_size;
    std::generate(input_data.begin(), input_data.end(), [&current_value]() { return current_value--; });
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    task_data->inputs_count.emplace_back(input_data.size());

    output_data.resize(vector_size);
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_data.data()));
    task_data->outputs_count.emplace_back(output_data.size());
  }

  auto parallel_sort_task =
      std::make_shared<yasakova_t_quick_sort_with_simple_merge_mpi::SimpleMergeQuicksort>(task_data);

  ASSERT_TRUE(parallel_sort_task->validation());
  parallel_sort_task->pre_processing();
  parallel_sort_task->run();
  parallel_sort_task->post_processing();

  auto performance_attributes = std::make_shared<ppc::core::PerfAttr>();
  performance_attributes->num_running = 10;
  const boost::mpi::timer timer;
  performance_attributes->current_timer = [&] { return timer.elapsed(); };

  auto performance_results = std::make_shared<ppc::core::PerfResults>();

  auto performance_analyzer = std::make_shared<ppc::core::Perf>(parallel_sort_task);
  performance_analyzer->task_run(performance_attributes, performance_results);

  if (mpi_comm.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(performance_results);
    std::sort(input_data.begin(), input_data.end());
    EXPECT_EQ(input_data, output_data);
  }
}