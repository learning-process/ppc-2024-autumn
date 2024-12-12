#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <memory>
#include <vector>

#include "../include/ops_mpi.hpp"
#include "core/perf/include/perf.hpp"

void run_test(std::string &&in, size_t solution,
              const std::function<void(ppc::core::Perf &, const std::shared_ptr<ppc::core::PerfAttr>,
                                       const std::shared_ptr<ppc::core::PerfResults>)> &executor) {
  boost::mpi::communicator world;

  size_t out = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<vedernikova_k_word_num_in_str_mpi::TestMPITaskParallel>(taskDataPar);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  ppc::core::Perf perfAnalyzer(testMpiTaskParallel);
  executor(perfAnalyzer, perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_EQ(out, solution);
  }
}
void run_test(const std::function<void(ppc::core::Perf &, const std::shared_ptr<ppc::core::PerfAttr>,
                                       const std::shared_ptr<ppc::core::PerfResults>)> &executor) {
  run_test("Sentence for word counter test", 5, executor);
}

TEST(vedernikova_k_word_num_in_str_mpi_perf_test, test_pipeline_run) {
  run_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.pipeline_run(perfAttr, perfResults);
  });
}

TEST(vedernikova_k_word_num_in_str_mpi_perf_test, test_task_run) {
  run_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.task_run(perfAttr, perfResults);
  });
}
