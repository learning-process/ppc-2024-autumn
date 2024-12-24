#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "../include/ops_mpi.hpp"
#include "core/perf/include/perf.hpp"

template <typename DataType>
static std::vector<DataType> generateRandomValues(int size) {
  std::vector<DataType> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = static_cast<DataType>(rand() % 100);
  }
  return vec;
}

TEST(moiseev_a_radix_merge_mpi_perf, test_pipeline_run) {
  boost::mpi::communicator world;

  using DataType = int32_t;
  const size_t vector_size = 1000000;
  std::vector<DataType> in = generateRandomValues<DataType>(vector_size);
  std::vector<DataType> out(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto task = std::make_shared<moiseev_a_radix_merge_mpi::TestMPITaskParallel>(taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    std::vector<DataType> ref = in;
    std::sort(ref.begin(), ref.end());
    ASSERT_EQ(out, ref);

    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(moiseev_a_radix_merge_mpi_perf, test_task_run) {
  boost::mpi::communicator world;

  using DataType = int32_t;
  const size_t vector_size = 1000000;
  std::vector<DataType> in = generateRandomValues<DataType>(vector_size);
  std::vector<DataType> out(vector_size, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto task = std::make_shared<moiseev_a_radix_merge_mpi::TestMPITaskParallel>(taskData);
  ASSERT_EQ(task->validation(), true);
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    std::vector<DataType> ref = in;
    std::sort(ref.begin(), ref.end());
    ASSERT_EQ(out, ref);

    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
