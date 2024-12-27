#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/sharamygina_i_horizontal_line_filtration/include/ops_mpi.hpp"

using namespace sharamygina_i_horizontal_line_filtration_mpi {
  std::vector<unsigned int> GetImage(int rows, int cols) {
    std::vector<unsigned int> temporaryIm(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, std::numeric_limits<unsigned int>::max());
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) temporaryIm[i * cols + j] = dist(gen);
    return temporaryIm;
  }
}  // namespace sharamygina_i_horizontal_line_filtration_mpi

#define PERF_TEST_IMAGE(test_name, R, C, num_runs, perf_method)
TEST(sharamygina_i_horizontal_line_filtration, PerfTest) {
  boost::mpi::communicator world;
  int rows = (R);
  int cols = (C);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> image;
  std::vector<int> received_image;
  if (world.rank() == 0) {
    image = generate_random_image(rows, cols);
    received_image.resize(rows * cols, 0);
    taskData->inputs.push_back(reinterpret_cast<uint8_t *>(image.data()));
    taskData->inputs_count.push_back(rows);
    taskData->inputs_count.push_back(cols);

    taskData->outputs.push_back(reinterpret_cast<uint8_t *>(received_image.data()));
    taskData->outputs_count.push_back(received_image.size() * sizeof(int));
  }
  auto task = std::make_shared<anufriev_d_linear_image::SimpleIntMPI>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = num_runs;
  boost::mpi::timer current_timer;
  perfAttr->current_timer = [&]() { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->perf_method(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_LE(perfResults->time_sec, ppc::core::PerfResults::MAX_TIME);
  }
}

PERF_TEST_IMAGE(LargeImagePerf, 5000, 5000, 1, pipeline_run)

PERF_TEST_IMAGE(LargeImageTaskRunPerf, 5000, 5000, 1, task_run)

#undef PERF_TEST_IMAGE