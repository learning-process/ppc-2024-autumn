#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/rams_s_radix_sort_with_simple_merge_for_doubles/include/ops_mpi.hpp"

void rams_s_radix_sort_with_simple_merge_for_doubles_mpi_run_perf_test(bool pipeline) {
  boost::mpi::communicator world;
  size_t length = 500000;
  std::vector<double> in;
  std::vector<double> out;
  std::vector<double> expected;

  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<double>(length, 0);
    std::random_device dev;
    std::mt19937_64 gen(dev());
    for (size_t i = 0; i < length; i++) {
      while (std::isnan(in[i] = std::bit_cast<double>(gen())));
    }
    out = std::vector<double>(length, 0);
    expected = std::vector<double>(in);
    std::sort(expected.begin(), expected.end());
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data->inputs_count.emplace_back(in.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  auto test_task =
      std::make_shared<rams_s_radix_sort_with_simple_merge_for_doubles_mpi::TestMPITaskParallel>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test_task);
  if (pipeline) {
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
  } else {
    perfAnalyzer->task_run(perfAttr, perfResults);
  }
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected, out);
  }
}

TEST(rams_s_radix_sort_with_simple_merge_for_doubles_mpi_perf_test, test_pipeline_run) {
  rams_s_radix_sort_with_simple_merge_for_doubles_mpi_run_perf_test(true);
}

TEST(rams_s_radix_sort_with_simple_merge_for_doubles_mpi_perf_test, test_task_run) {
  rams_s_radix_sort_with_simple_merge_for_doubles_mpi_run_perf_test(false);
}
