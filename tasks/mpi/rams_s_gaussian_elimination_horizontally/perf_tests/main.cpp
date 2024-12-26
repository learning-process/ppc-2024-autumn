#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <cmath>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/rams_s_gaussian_elimination_horizontally/include/ops_mpi.hpp"

void rams_s_gaussian_elimination_horizontally_seq_run_perf_test(bool pipeline) {
  boost::mpi::communicator world;
  size_t cols_count = 251;
  size_t rows_count = cols_count - 1;
  std::vector<double> in(rows_count * cols_count, 0);
  for (size_t row = 0; row < rows_count; row++) {
    double sum = 0;
    size_t col = 0;
    for (; col < cols_count - 1; col++) {
      sum += (in[row * cols_count + col] = std::min(col, row) + 1);
    }
    in[row * cols_count + col] = -sum;
  }
  std::vector<double> out(rows_count, 0);
  std::vector<double> expected(rows_count, 1);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<rams_s_gaussian_elimination_horizontally_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  if (pipeline) {
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  } else {
    perfAnalyzer->task_run(perfAttr, perfResults);
  }
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < expected.size(); i++) {
      if (std::isnan(expected[i])) {
        ASSERT_TRUE(std::isnan(out[i]));
      } else {
        ASSERT_DOUBLE_EQ(expected[i], out[i]);
      }
    }
  }
}

TEST(rams_s_gaussian_elimination_horizontally_seq_perf_test, test_pipeline_run) {
  rams_s_gaussian_elimination_horizontally_seq_run_perf_test(true);
}

TEST(rams_s_gaussian_elimination_horizontally_seq_perf_test, test_task_run) {
  rams_s_gaussian_elimination_horizontally_seq_run_perf_test(false);
}
