#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/laganina_e_sum_values_by_columns_matrix/include/ops_seq.hpp"
TEST(laganina_e_sum_values_by_columns_matrix_seq, test_pipeline_run) {
    int n = 5000;
    int m = 5000;
    // Create data
    std::vector<int> in(n*m, 1);
    std::vector<int> empty(n, 0);
    std::vector<int> out(n,m);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    // taskDataSeq->inputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty.data()));
    taskDataSeq->outputs_count.emplace_back(empty.size());
    // Create Task
    auto testTaskSequential =
    std::make_shared<laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq>(taskDataSeq);
    // Create Perf attributes
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
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(empty, out);
}

TEST(laganina_e_sum_values_by_columns_matrix_seq, test_task_run) {
    int n = 5000;
    int m = 5000;
    // Create data
    std::vector<int> in(n * m, 1);
    std::vector<int> empty(n, 0);
    std::vector<int> out(n, m);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(m);
    taskDataSeq->inputs_count.emplace_back(n);
    // taskDataSeq->inputs_count.emplace_back((size_t)1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty.data()));
    taskDataSeq->outputs_count.emplace_back(empty.size());
    // Create Task
    auto testTaskSequential = std::make_shared<laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq>(taskDataSeq);
    // Create Perf attributes
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
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
    perfAnalyzer->task_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(empty, out);
}
