#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include <random>
#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"

namespace nasedkin_e_seidels_iterate_methods_mpi {

    std::vector<double> genRandomMatrix(int n, int m) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        std::vector<double> matrix(n * m);
        for (int i = 0; i < n * m; i++) {
            matrix[i] = dis(gen);
        }
        return matrix;
    }

    std::vector<double> genRandomVector(int n) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        std::vector<double> vec(n);
        for (int i = 0; i < n; i++) {
            vec[i] = dis(gen);
        }
        return vec;
    }

    TEST(MPISeidelPerf, TestPipelineRun) {
        boost::mpi::communicator world;
        int rows = 1000;
        int columns = 1000;
        std::vector<double> matrix = genRandomMatrix(rows, columns);
        std::vector<double> b = genRandomVector(rows);
        std::vector<double> expres_par(rows);

        std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
        if (world.rank() == 0) {
            taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
            taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
            taskDataPar->inputs_count.emplace_back(matrix.size());
            taskDataPar->inputs_count.emplace_back(b.size());
            taskDataPar->inputs_count.emplace_back(columns);
            taskDataPar->inputs_count.emplace_back(rows);
            taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_par.data()));
            taskDataPar->outputs_count.emplace_back(expres_par.size());
        }

        auto testMpiTaskParallel = std::make_shared<TestMPITaskParallel>(taskDataPar);
        ASSERT_EQ(testMpiTaskParallel->validation(), true);
        testMpiTaskParallel->pre_processing();
        testMpiTaskParallel->run();
        testMpiTaskParallel->post_processing();

        auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
        perfAttr->num_running = 10;
        const boost::mpi::timer current_timer;
        perfAttr->current_timer = [&] { return current_timer.elapsed(); };

        auto perfResults = std::make_shared<ppc::core::PerfResults>();

        auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
        perfAnalyzer->pipeline_run(perfAttr, perfResults);
        if (world.rank() == 0) {
            ppc::core::Perf::print_perf_statistic(perfResults);
        }
    }

    TEST(MPISeidelPerf, TestTaskRun) {
        boost::mpi::communicator world;
        int rows = 1000;
        int columns = 1000;
        std::vector<double> matrix = genRandomMatrix(rows, columns);
        std::vector<double> b = genRandomVector(rows);
        std::vector<double> expres_par(rows);

        std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
        if (world.rank() == 0) {
            taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
            taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
            taskDataPar->inputs_count.emplace_back(matrix.size());
            taskDataPar->inputs_count.emplace_back(b.size());
            taskDataPar->inputs_count.emplace_back(columns);
            taskDataPar->inputs_count.emplace_back(rows);
            taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_par.data()));
            taskDataPar->outputs_count.emplace_back(expres_par.size());
        }

        auto testMpiTaskParallel = std::make_shared<TestMPITaskParallel>(taskDataPar);
        ASSERT_EQ(testMpiTaskParallel->validation(), true);
        testMpiTaskParallel->pre_processing();
        testMpiTaskParallel->run();
        testMpiTaskParallel->post_processing();

        auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
        perfAttr->num_running = 10;
        const boost::mpi::timer current_timer;
        perfAttr->current_timer = [&] { return current_timer.elapsed(); };

        auto perfResults = std::make_shared<ppc::core::PerfResults>();

        auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
        perfAnalyzer->task_run(perfAttr, perfResults);
        if (world.rank() == 0) {
            ppc::core::Perf::print_perf_statistic(perfResults);
        }
    }

}  // namespace nasedkin_e_seidels_iterate_methods_mpi