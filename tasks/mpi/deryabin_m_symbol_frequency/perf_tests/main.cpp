#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/deryabin_m_symbol_frequency/include/ops_mpi.hpp"

#include <string>

TEST(deryabin_m_symbol_frequency_mpi, test_pipeline_run) {
    const float TEST_frequency = 1;
    boost::mpi::communicator world;
    std::vector<char> global_str;
    std::vector<char> input_char(1, '@');
    std::vector<float> global_frequency(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    int count_size_str;
    if (world.rank() == 0) {
        count_size_str = 10000;
        global_str = std::vector<char>(count_size_str, input_char[0]);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
        taskDataPar->inputs_count.emplace_back(global_str.size());
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_char.data()));
        taskDataPar->inputs_count.emplace_back(input_char.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
        taskDataPar->outputs_count.emplace_back(global_frequency.size());
    }

    auto testMpiTaskParallel = std::make_shared<deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel>(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
        ASSERT_EQ(TEST_frequency, global_frequency[0]);
    }
}

TEST(deryabin_m_symbol_frequency_mpi, test_task_run) {
    const float TEST_frequency = 1;
    boost::mpi::communicator world;
    std::vector<char> global_str;
    std::vector<char> input_char(1, '@');
    std::vector<float> global_frequency(1, 0);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    int count_size_str;
    if (world.rank() == 0) {
        count_size_str = 10000;
        global_str = std::vector<char>(count_size_str, input_char[0]);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
        taskDataPar->inputs_count.emplace_back(global_str.size());
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_char.data()));
        taskDataPar->inputs_count.emplace_back(input_char.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
        taskDataPar->outputs_count.emplace_back(global_frequency.size());
    }

    auto testMpiTaskParallel = std::make_shared<deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel>(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();
    testMpiTaskParallel->run();
    testMpiTaskParallel->post_processing();

    // Create Perf attributes
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    // Create and init perf results
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    // Create Perf analyzer
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->task_run(perfAttr, perfResults);
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
        ASSERT_EQ(TEST_frequency, global_frequency[0]);
    }
}
