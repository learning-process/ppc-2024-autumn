#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/lavrentyev_a_radix_sort_simple_merge/include/ops_mpi.hpp"

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, pipeline_run) {
  boost::mpi::communicator world;
  int vector_size = 1000000;
  std::vector<double> data;
  std::vector<double> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(vector_size);

  if (world.rank() == 0) {
    data.resize(vector_size);
    double current = vector_size;
    std::generate(data.begin(), data.end(), [&current]() { return current--; });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));

    result_data.resize(vector_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  }

  auto taskParallel = std::make_shared<lavrentyev_a_radix_sort_simple_merge_mpi::RadixSimpleMerge>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}

TEST(lavrentyev_a_radix_sort_simple_merge_mpi, task_run) {
  boost::mpi::communicator world;
  int vector_size = 1000000;
  std::vector<double> data;
  std::vector<double> result_data;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(vector_size);

  if (world.rank() == 0) {
    data.resize(vector_size);
    double current = vector_size;
    std::generate(data.begin(), data.end(), [&current]() { return current--; });

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));

    result_data.resize(vector_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  }

  auto taskParallel = std::make_shared<lavrentyev_a_radix_sort_simple_merge_mpi::RadixSimpleMerge>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::sort(data.begin(), data.end());
    EXPECT_EQ(data, result_data);
  }
}
