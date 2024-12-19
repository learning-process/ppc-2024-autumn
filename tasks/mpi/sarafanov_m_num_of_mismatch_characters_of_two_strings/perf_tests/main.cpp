#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sarafanov_m_num_of_mismatch_characters_of_two_strings/include/ops_mpi.hpp"

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::string input_a;
  std::string input_b;

  auto output = std::vector<int>(1);
  auto expected = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_a = std::string(5000000, 'a');
    input_b = std::string(5000000, 'a');
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
    task_data->inputs_count.emplace_back(input_a.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
    task_data->inputs_count.emplace_back(input_b.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::ParallelTask>(task_data);
  ASSERT_TRUE(task->validation());
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->pipeline_run(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    ASSERT_EQ(expected, output[0]);
  }
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::string input_a;
  std::string input_b;

  auto output = std::vector<int>(1);
  auto expected = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    input_a = std::string(5000000, 'a');
    input_b = std::string(5000000, 'a');
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
    task_data->inputs_count.emplace_back(input_a.size());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
    task_data->inputs_count.emplace_back(input_b.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
    task_data->outputs_count.emplace_back(output.size());
  }

  auto task = std::make_shared<sarafanov_m_num_of_mismatch_characters_of_two_strings_mpi::ParallelTask>(task_data);
  ASSERT_TRUE(task->validation());
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->task_run(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);
    ASSERT_EQ(expected, output[0]);
  }
}
