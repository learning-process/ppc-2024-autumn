#include <gtest/gtest.h>
#include "core/perf/include/perf.hpp"
#include "seq/nasedkin_e_seidels_iterate_methods/include/ops_seq.hpp"

TEST(nasedkin_e_seidels_iterate_methods_seq, test_pipeline_run) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(3);

    auto seidelTask = std::make_shared<nasedkin_e_seidels_iterate_methods_seq::SeidelIterateMethodsSeq>(taskData);

    ASSERT_TRUE(seidelTask->validation()) << "Validation failed for valid input";

    seidelTask->pre_processing();
    seidelTask->run();
    seidelTask->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;  // Количество запусков для теста
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(seidelTask);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    ppc::core::Perf::print_perf_statistic(perfResults);
}
