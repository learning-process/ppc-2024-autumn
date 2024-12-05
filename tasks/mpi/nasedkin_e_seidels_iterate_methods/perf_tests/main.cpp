#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"

namespace nasedkin_e_seidels_iterate_methods_mpi {
    template <typename T>
    std::vector<T> getRandomVector(int sz) {
        std::random_device dev;
        std::mt19937 gen(dev());
        std::vector<T> vec(sz);
        vec[0] = gen() % 100;
        for (int i = 1; i < sz; i++) {
            vec[i] = (gen() % 100) - 49;
        }
        return vec;
    }

    template std::vector<int> nasedkin_e_seidels_iterate_methods_mpi::getRandomVector(int sz);
    template std::vector<double> nasedkin_e_seidels_iterate_methods_mpi::getRandomVector(int sz);
}  // namespace nasedkin_e_seidels_iterate_methods_mpi

TEST(MPISeidelPerf, test_pipeline_run) {
    boost::mpi::communicator world;
    int rows = 900;
    int columns = 900;
    std::vector<int> a = nasedkin_e_seidels_iterate_methods_mpi::getRandomVector<int>(1);
    std::vector<double> matrix = nasedkin_e_seidels_iterate_methods_mpi::generateDenseMatrix(rows, *a.begin());
    std::vector<double> b(rows, 1);
    std::vector<double> res(rows, 0);
    res[0] = -1;
    res[1] = 1;
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

    auto testMpiTaskParallel = std::make_shared<nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel>(taskDataPar);
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
        ASSERT_EQ(expres_par, res);
    }
}

TEST(MPISeidelPerf, test_task_run) {
    boost::mpi::communicator world;
    int rows = 900;
    int columns = 900;
    std::vector<int> a = nasedkin_e_seidels_iterate_methods_mpi::getRandomVector<int>(1);
    std::vector<double> matrix = nasedkin_e_seidels_iterate_methods_mpi::generateDenseMatrix(rows, *a.begin());
    std::vector<double> b(rows, 1);
    std::vector<double> res(rows, 0);
    res[0] = -1;
    res[1] = 1;
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

    auto testMpiTaskParallel = std::make_shared<nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel>(taskDataPar);
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
        ASSERT_EQ(expres_par, res);
    }
}