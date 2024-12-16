#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"
#include "mpi/nasedkin_e_strassen_algorithm/src/ops_mpi.cpp"

TEST(nasedkin_e_strassen_algorithm_mpi, test_pipeline_run) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(8);

    auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskData);

    ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

    strassenTask->pre_processing();
    strassenTask->run();
    strassenTask->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTask);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_task_run) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(8);

    auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI>(taskData);

    ASSERT_TRUE(strassenTask->validation()) << "Validation failed for valid input";

    strassenTask->pre_processing();
    strassenTask->run();
    strassenTask->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(strassenTask);
    perfAnalyzer->task_run(perfAttr, perfResults);

    ppc::core::Perf::print_perf_statistic(perfResults);
}