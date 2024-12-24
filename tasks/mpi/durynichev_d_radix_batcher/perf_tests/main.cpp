#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/durynichev_d_radix_batcher/include/ops_mpi.hpp"

TEST(durynichev_d_radix_batcher_mpi, pipeline_run) {
  boost::mpi::communicator world;
  size_t array_size = 262144;
  std::vector<double> array;
  std::vector<double> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&array_size));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    array.resize(array_size);
    double current = array_size;
    std::generate(array.begin(), array.end(), [&current]() { return current--; });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataPar->inputs_count.emplace_back(array.size());

    result.resize(array_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  auto taskParallel = std::make_shared<durynichev_d_radix_batcher_mpi::RadixBatcher>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(array.begin(), array.end());
    EXPECT_EQ(array, result);
  }
}

TEST(durynichev_d_radix_batcher_mpi, task_run) {
  boost::mpi::communicator world;
  size_t array_size = 262144;
  std::vector<double> array;
  std::vector<double> result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&array_size));
  taskDataPar->inputs_count.emplace_back(1);

  if (world.rank() == 0) {
    array.resize(array_size);
    double current = array_size;
    std::generate(array.begin(), array.end(), [&current]() { return current--; });
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskDataPar->inputs_count.emplace_back(array.size());

    result.resize(array_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskDataPar->outputs_count.emplace_back(result.size());
  }

  auto taskParallel = std::make_shared<durynichev_d_radix_batcher_mpi::RadixBatcher>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(array.begin(), array.end());
    EXPECT_EQ(array, result);
  }
}
