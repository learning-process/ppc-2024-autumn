#include <gtest/gtest.h>

#include <memory>
#include <random>

#include "../include/smmul_seq.hpp"
#include "core/perf/include/perf.hpp"

class krylov_m_crs_mmul_seq_test : public ::testing::Test {
  using TestElementType = double;
  static constexpr float Density = 0.10f;

  static constexpr size_t lrows = 411;
  static constexpr size_t lcols = 411;
  static constexpr size_t rcols = 411;
  //
  static constexpr TestElementType emin = -128;
  static constexpr TestElementType emax = 128;

 protected:
  static void run_perf_test(
      const std::function<void(ppc::core::Perf &perfAnalyzer, const std::shared_ptr<ppc::core::PerfAttr> &perfAttr,
                               const std::shared_ptr<ppc::core::PerfResults> &perfResults)> &runner) {
    const krylov_m_crs_mmul::Matrix<TestElementType> lhs =
        generate_random_matrix<TestElementType>(lrows, lcols, emin, emax, Density);
    const krylov_m_crs_mmul::Matrix<TestElementType> rhs =
        generate_random_matrix<TestElementType>(lcols, rcols, emin, emax, Density);
    //

    krylov_m_crs_mmul::CRSMatrix<TestElementType> slhs(lhs);
    krylov_m_crs_mmul::CRSMatrix<TestElementType> srhs(rhs);
    krylov_m_crs_mmul::CRSMatrix<TestElementType> sout;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    krylov_m_crs_mmul::fill_task_data(*taskData, slhs, srhs, sout);

    //
    auto task = std::make_shared<krylov_m_crs_mmul::TaskSequential<TestElementType>>(taskData);

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

  template <typename T>
  static krylov_m_crs_mmul::Matrix<T> generate_random_matrix(size_t rows, size_t cols, T emin, T emax, float density) {
    const T threshold = emin + ((emax - emin) * density);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> distr(emin, emax);

    auto matrix = krylov_m_crs_mmul::Matrix<T>::create(rows, cols);
    std::generate(matrix.storage.begin(), matrix.storage.end(), [&]() {
      auto val = distr(gen);
      return (val < threshold) ? val : 0;
    });

    return matrix;
  }
};

TEST_F(krylov_m_crs_mmul_seq_test, test_pipeline_run) {
  run_perf_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.pipeline_run(perfAttr, perfResults);
  });
}

TEST_F(krylov_m_crs_mmul_seq_test, test_task_run) {
  run_perf_test([](auto &perfAnalyzer, const auto &perfAttr, const auto &perfResults) {
    perfAnalyzer.task_run(perfAttr, perfResults);
  });
}