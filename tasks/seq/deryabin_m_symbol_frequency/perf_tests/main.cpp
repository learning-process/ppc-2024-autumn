#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/deryabin_m_symbol_frequency/include/ops_sec.hpp"

#include <string>

TEST(deryabin_m_symbol_frequency_seq, test_pipeline_run) {
    const double TEST_frequency = 1;

    // Create data
    std::vector<std::string> in(1, std::string(10000, '@'));
    std::vector<char> in_ch(1, '@');
    std::vector<double> out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_ch.data()));
    taskDataSeq->inputs_count.emplace_back(in_ch.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    // Create Task
    auto symbol_frequency_TaskSequential = std::make_shared<deryabin_m_symbol_frequency_seq::Symbol_frequency_TaskSequential>(taskDataSeq);

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
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(symbol_frequency_TaskSequential);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(TEST_frequency, out[0]);
}

TEST(deryabin_m_symbol_frequency_seq, test_task_run) {
    const double TEST_frequency = 1;

    // Create data
    std::vector<std::string> in(1, std::string(10000, '@'));
    std::vector<char> in_ch(1, '@');
    std::vector<double> out(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_ch.data()));
    taskDataSeq->inputs_count.emplace_back(in_ch.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataSeq->outputs_count.emplace_back(out.size());

    // Create Task
    auto symbol_frequency_TaskSequential = std::make_shared<deryabin_m_symbol_frequency_seq::Symbol_frequency_TaskSequential>(taskDataSeq);

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
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(symbol_frequency_TaskSequential);
    perfAnalyzer->task_run(perfAttr, perfResults);
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(TEST_frequency, out[0]);
}
