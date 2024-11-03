#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>
#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

TEST(MatrixColumnMaxPerfTest, ParallelPerformanceTest) {
    boost::mpi::communicator world;
    int size = 100000;
    std::vector<int> vec;
    std::vector<int32_t> output(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    if (world.rank() == 0) {
        vec = nasedkin_e_matrix_column_max_value_mpi::getRandomVector(size);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
        taskDataPar->inputs_count.emplace_back(vec.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
        taskDataPar->outputs_count.emplace_back(output.size());
    }

    auto parTask =
        std::make_shared<nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxTaskParallel>(taskDataPar, "max");
    ASSERT_TRUE(parTask->validation());

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };
    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parTask);
    perfAnalyzer->task_run(perfAttr, perfResults);
    if (world.rank() == 0) {
        ppc::core::Perf::print_perf_statistic(perfResults);
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
