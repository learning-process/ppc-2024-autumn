#include <gtest/gtest.h>

#include "boost/mpi/communicator.hpp"
#include "boost/mpi/timer.hpp"
#include "core/perf/include/perf.hpp"
#include "mpi/agafeev_s_max_of_vector_elements/include/ops_mpi.hpp"

template <typename T>
static std::vector<T> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(std::time(nullptr));
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); ++i) matrix[i] = rand_gen() % 200 - 100;

  return matrix;
}

TEST(agafeev_s_max_of_vector_elements, test_pipeline_run) {
  const int n = 4000;
  const int m = 4000;
  boost::mpi::communicator world;
  // Credate Data
  std::vector<int> in_matrix(n * m);
  std::vector<int> out(1, 99);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<int>(n, m);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTaskMpi = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer;
  perfAttr->current_timer = [&] { return timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMpi);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto temp = agafeev_s_max_of_vector_elements_mpi::get_MaxValue<int>(in_matrix);
    ASSERT_EQ(temp, out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_task_run) {
  const int n = 4000;
  const int m = 4000;
  boost::mpi::communicator world;
  // Credate Data
  std::vector<int> in_matrix(n * m);
  std::vector<int> out(1, 99);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<int>(n, m);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTaskMpi = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer;
  perfAttr->current_timer = [&] { return timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskMpi);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto temp = agafeev_s_max_of_vector_elements_mpi::get_MaxValue<int>(in_matrix);
    ASSERT_EQ(temp, out[0]);
  }
}
