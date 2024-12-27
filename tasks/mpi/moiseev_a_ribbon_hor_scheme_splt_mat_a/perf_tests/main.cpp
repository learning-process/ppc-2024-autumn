#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/moiseev_a_ribbon_hor_scheme_splt_mat_a/include/ops_mpi.hpp"
#include "seq/moiseev_a_ribbon_hor_scheme_splt_mat_a/include/ops_seq.hpp"

template <typename DataType>
static std::vector<DataType> generateRandomValues(int size) {
  std::vector<DataType> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = static_cast<DataType>(rand() % 100);
  }
  return vec;
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_pipeline_run) {
  boost::mpi::communicator world;

  using DataType = int32_t;
  const size_t m = 100;
  const size_t k = 100;
  const size_t n = 100;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C(m * n, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->inputs_count.emplace_back(m);
    taskData->inputs_count.emplace_back(k);
    taskData->inputs_count.emplace_back(n);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
    taskData->outputs_count.emplace_back(C.size());
  }

  auto task =
      std::make_shared<moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType>>(taskData);
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
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(C.size(), m * n);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, test_task_run) {
  boost::mpi::communicator world;

  using DataType = int32_t;
  const size_t m = 100;
  const size_t k = 100;
  const size_t n = 100;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C(m * n, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskData->inputs_count.emplace_back(m);
    taskData->inputs_count.emplace_back(k);
    taskData->inputs_count.emplace_back(n);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
    taskData->outputs_count.emplace_back(C.size());
  }

  auto task =
      std::make_shared<moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType>>(taskData);
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
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(C.size(), m * n);
  }
}
