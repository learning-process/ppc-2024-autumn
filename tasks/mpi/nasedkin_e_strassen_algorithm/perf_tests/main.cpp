#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "../include/ops_mpi.hpp"

TEST(mpi_strassen_perf_test, test_pipeline_run) {
    boost::mpi::communicator world;
    std::vector<int> A, B;
    std::vector<int> C(128 * 128, 0);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        A = std::vector<int>(128 * 128, 1);
        B = std::vector<int>(128 * 128, 1);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
        taskData->inputs_count.emplace_back(A.size());
        taskData->inputs_count.emplace_back(B.size());
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
        taskData->outputs_count.emplace_back(C.size());
    }

    auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenMPITaskParallel>(taskData);
    ASSERT_EQ(strassenTask->validation(), true);
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

    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
    }
}

TEST(mpi_strassen_perf_test, test_task_run) {
    boost::mpi::communicator world;
    std::vector<int> A, B;
    std::vector<int> C(128 * 128, 0);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        A = std::vector<int>(128 * 128, 1);
        B = std::vector<int>(128 * 128, 1);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
        taskData->inputs_count.emplace_back(A.size());
        taskData->inputs_count.emplace_back(B.size());
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
        taskData->outputs_count.emplace_back(C.size());
    }

    auto strassenTask = std::make_shared<nasedkin_e_strassen_algorithm::StrassenMPITaskParallel>(taskData);
    ASSERT_EQ(strassenTask->validation(), true);
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

    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
    }
}