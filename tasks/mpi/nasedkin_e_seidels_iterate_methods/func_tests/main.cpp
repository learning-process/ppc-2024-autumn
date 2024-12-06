#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"

/*
TEST(MPISeidel, ZeroDiagonalTest) {
    boost::mpi::communicator world;
    int rows = 3;
    int columns = 3;
    std::vector<double> matrix = {0, 1, 1, 1, 0, 1, 1, 1, 0};
    std::vector<double> b = {1, 1, 1};
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

    nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
}
*/


TEST(MPISeidel, RandomMatrixTest) {
    boost::mpi::communicator world;
    int rows = 10;
    int columns = 10;
    std::vector<double> matrix = nasedkin_e_seidels_iterate_methods_mpi::generateDenseMatrix(rows, 1);
    std::vector<double> b = nasedkin_e_seidels_iterate_methods_mpi::getRandomVector<double>(rows);
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

    nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
        std::vector<double> Ax(rows, 0.0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                Ax[i] += matrix[i * columns + j] * expres_par[j];
            }
        }

        double norm = 0.0;
        for (int i = 0; i < rows; ++i) {
            norm += std::abs(Ax[i] - b[i]);
        }

        ASSERT_LT(norm, 1e-6);
    }
}

TEST(MPISeidel, IdentityMatrixTest) {
    boost::mpi::communicator world;
    int rows = 10;
    int columns = 10;
    std::vector<double> matrix = nasedkin_e_seidels_iterate_methods_mpi::generateElementaryMatrix(rows, columns);
    std::vector<double> b(rows, 1);
    std::vector<double> res_par(rows, 1);
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

    nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
        ASSERT_EQ(expres_par, res_par);
    }
}

TEST(MPISeidel, LargeMatrixTest) {
    boost::mpi::communicator world;
    int rows = 100;
    int columns = 100;
    std::vector<double> matrix = nasedkin_e_seidels_iterate_methods_mpi::generateDenseMatrix(rows, 1);
    std::vector<double> b(rows, 1);
    std::vector<double> res_par(rows, 0);
    res_par[0] = -1;
    res_par[1] = 1;
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

    nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
        ASSERT_EQ(expres_par, res_par);
    }
}

TEST(MPISeidel, EmptyMatrixTest) {
    boost::mpi::communicator world;
    int rows = 0;
    int columns = 0;
    std::vector<double> matrix = {};
    std::vector<double> b = {};
    std::vector<double> expres_par(rows, 0);
    std::vector<double> res_par = {};

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

    nasedkin_e_seidels_iterate_methods_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
        ASSERT_EQ(expres_par, res_par);
    }
}