#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "../include/ops_mpi.hpp"
#include "core/perf/include/perf.hpp"

class krylov_m_num_of_alternations_signs_mpi_perf_test : public ::testing::Test {
  using ElementType = int32_t;
  using CountType = uint32_t;
  //
  const CountType in_count = 128;
  const std::vector<CountType> shift_indices{0, 1, /* . */ 3, /* . */ 5, 6, 7, /* . */ 12 /* . */};
  //
  const CountType num = 7;

 protected:
  void run_perf_test(
      const std::function<void(ppc::core::Perf &perfAnalyzer, const std::shared_ptr<ppc::core::PerfAttr> &perfAttr,
                               const std::shared_ptr<ppc::core::PerfResults> &perfResults)> &runner) {
    boost::mpi::communicator world;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    //
    std::vector<ElementType> in;
    CountType out = 0;
    if (world.rank() == 0) {
      in = std::vector<ElementType>(in_count);
      std::iota(in.begin(), in.end(), 1);

      for (auto idx : shift_indices) {
        in[idx] *= -1;
      }

      //
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      taskDataPar->inputs_count.emplace_back(in.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
      taskDataPar->outputs_count.emplace_back(1);
    }

    //
    auto testMpiTaskParallel =
        std::make_shared<krylov_m_num_of_alternations_signs_mpi::TestMPITaskParallel<ElementType, CountType>>(
            taskDataPar);

    //
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    ppc::core::Perf perfAnalyzer(testMpiTaskParallel);
    runner(perfAnalyzer, perfAttr, perfResults);
    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      EXPECT_EQ(num, out);
    }
  }
};

TEST_F(krylov_m_num_of_alternations_signs_mpi_perf_test, test_pipeline_run) {
  run_perf_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.pipeline_run(perfAttr, perfResults);
  });
}

TEST_F(krylov_m_num_of_alternations_signs_mpi_perf_test, test_task_run) {
  run_perf_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.task_run(perfAttr, perfResults);
  });
}
