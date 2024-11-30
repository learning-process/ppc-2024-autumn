#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/lopatin_i_strip_horizontal_scheme/include/stripHorizontalSchemeHeaderMPI.hpp"

std::vector<int> testMatrix = lopatin_i_strip_horizontal_scheme_mpi::generateMatrix(3840, 2160);
std::vector<int> testVector = lopatin_i_strip_horizontal_scheme_mpi::generateVector(3840);

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = testMatrix;
  std::vector<int> inputVector = testVector;
  std::vector<int> resultVector(2160, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(6);
    taskDataParallel->inputs_count.emplace_back(4);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  auto testTask = std::make_shared<lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel>(taskDataParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(lopatin_i_strip_horizontal_scheme_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> inputMatrix = testMatrix;
  std::vector<int> inputVector = testVector;
  std::vector<int> resultVector(2160, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
    taskDataParallel->inputs_count.emplace_back(6);
    taskDataParallel->inputs_count.emplace_back(4);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
    taskDataParallel->inputs_count.emplace_back(inputVector.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
    taskDataParallel->outputs_count.emplace_back(resultVector.size());
  }

  auto testTask = std::make_shared<lopatin_i_strip_horizontal_scheme_mpi::TestMPITaskParallel>(taskDataParallel);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}