#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

namespace shuravina_o_contrast {
std::vector<uint8_t> genRandomData(uint32_t size) {
  std::vector<uint8_t> buff(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (uint32_t i = 0; i < size; i++) {
    buff[i] = gen() % 256;
  }
  return buff;
}
}  // namespace shuravina_o_contrast

TEST(shuravina_o_contrast_perf, test_pipeline_run_small_input) {
  boost::mpi::communicator world;
  std::vector<uint8_t> input;
  std::vector<uint8_t> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 1000;
    input = shuravina_o_contrast::genRandomData(size);
    res.resize(input.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto contrastTaskParallel = std::make_shared<shuravina_o_contrast::ContrastTaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(shuravina_o_contrast_perf, test_pipeline_run_large_input) {
  boost::mpi::communicator world;
  std::vector<uint8_t> input;
  std::vector<uint8_t> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 1000000;
    input = shuravina_o_contrast::genRandomData(size);
    res.resize(input.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  auto contrastTaskParallel = std::make_shared<shuravina_o_contrast::ContrastTaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}