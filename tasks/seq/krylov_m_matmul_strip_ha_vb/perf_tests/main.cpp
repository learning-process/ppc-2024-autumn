#include <gtest/gtest.h>

#include <memory>
#include <random>

#include "../include/mmul_seq.hpp"
#include "core/perf/include/perf.hpp"

class krylov_m_matmul_strip_ha_vb_seq_test : public ::testing::Test {
  using TestElementType = int64_t;

  static constexpr size_t lrows = 256;
  static constexpr size_t lcols = 256;
  static constexpr size_t rcols = 256;
  //
  static constexpr TestElementType emin = -64;
  static constexpr TestElementType emax = 64;

 protected:
  static void run_perf_test(
      const std::function<void(ppc::core::Perf &perfAnalyzer, const std::shared_ptr<ppc::core::PerfAttr> &perfAttr,
                               const std::shared_ptr<ppc::core::PerfResults> &perfResults)> &runner) {
    krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> lhs;
    krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> rhs;
    //
    krylov_m_matmul_strip_ha_vb_seq::TMatrix<TestElementType> out;

    auto taskData = std::make_shared<ppc::core::TaskData>();

    lhs = generate_random_matrix<TestElementType>(lrows, lcols, emin, emax);
    ASSERT_TRUE(lhs.check_integrity());
    //
    rhs = generate_random_matrix<TestElementType>(lcols, rcols, emin, emax);
    ASSERT_TRUE(rhs.check_integrity());

    krylov_m_matmul_strip_ha_vb_seq::fill_task_data(*taskData, lhs, rhs, out);

    //
    auto task = std::make_shared<krylov_m_matmul_strip_ha_vb_seq::TaskSequential<TestElementType>>(taskData);

    //
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const auto t0 = std::chrono::high_resolution_clock::now();
    perfAttr->current_timer = [&] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    //
    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    ppc::core::Perf perfAnalyzer(task);
    runner(perfAnalyzer, perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);
  }

  template <std::integral T>
  static krylov_m_matmul_strip_ha_vb_seq::TMatrix<T> generate_random_matrix(size_t rows, size_t cols, T min, T max) {
    auto matrix = krylov_m_matmul_strip_ha_vb_seq::TMatrix<T>::create(rows, cols);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> distr(min, max);

    std::generate(matrix.data.begin(), matrix.data.end(), [&distr, &gen]() { return distr(gen); });

    return matrix;
  }
};

TEST_F(krylov_m_matmul_strip_ha_vb_seq_test, test_pipeline_run) {
  run_perf_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.pipeline_run(perfAttr, perfResults);
  });
}

TEST_F(krylov_m_matmul_strip_ha_vb_seq_test, test_task_run) {
  run_perf_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.task_run(perfAttr, perfResults);
  });
}
