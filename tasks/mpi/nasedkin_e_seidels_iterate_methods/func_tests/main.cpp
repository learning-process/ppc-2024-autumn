#include <gtest/gtest.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>
#include <random>
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

    TEST(MPISeidel, TestRandomMatrix) {
        boost::mpi::communicator world;
        int rows = 10;
        int columns = 10;
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

        TestMPITaskParallel testMpiTaskParallel(taskDataPar);
        ASSERT_EQ(testMpiTaskParallel.validation(), true);
        testMpiTaskParallel.pre_processing();
        testMpiTaskParallel.run();
        testMpiTaskParallel.post_processing();

        if (world.rank() == 0) {
            std::vector<double> expres_seq(rows);
            std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
            taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
            taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
            taskDataSeq->inputs_count.emplace_back(matrix.size());
            taskDataSeq->inputs_count.emplace_back(b.size());
            taskDataSeq->inputs_count.emplace_back(columns);
            taskDataSeq->inputs_count.emplace_back(rows);
            taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_seq.data()));
            taskDataSeq->outputs_count.emplace_back(expres_seq.size());

            TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
            ASSERT_EQ(testMpiTaskSequential.validation(), true);
            testMpiTaskSequential.pre_processing();
            testMpiTaskSequential.run();
            testMpiTaskSequential.post_processing();

            ASSERT_EQ(expres_seq, expres_par);
        }
    }

    TEST(MPISeidel, TestZeroDiagonalMatrix) {
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

        TestMPITaskParallel testMpiTaskParallel(taskDataPar);
        ASSERT_EQ(testMpiTaskParallel.validation(), true);
        testMpiTaskParallel.pre_processing();
        testMpiTaskParallel.run();
        testMpiTaskParallel.post_processing();

        if (world.rank() == 0) {
            std::vector<double> expres_seq(rows);
            std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
            taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
            taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
            taskDataSeq->inputs_count.emplace_back(matrix.size());
            taskDataSeq->inputs_count.emplace_back(b.size());
            taskDataSeq->inputs_count.emplace_back(columns);
            taskDataSeq->inputs_count.emplace_back(rows);
            taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_seq.data()));
            taskDataSeq->outputs_count.emplace_back(expres_seq.size());

            TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
            ASSERT_EQ(testMpiTaskSequential.validation(), true);
            testMpiTaskSequential.pre_processing();
            testMpiTaskSequential.run();
            testMpiTaskSequential.post_processing();

            ASSERT_EQ(expres_seq, expres_par);
        }
    }

    TEST(MPISeidel, TestRandomMatrixWithCheck) {
        boost::mpi::communicator world;
        int rows = 10;
        int columns = 10;
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

        TestMPITaskParallel testMpiTaskParallel(taskDataPar);
        ASSERT_EQ(testMpiTaskParallel.validation(), true);
        testMpiTaskParallel.pre_processing();
        testMpiTaskParallel.run();
        testMpiTaskParallel.post_processing();

        if (world.rank() == 0) {
            std::vector<double> expres_seq(rows);
            std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
            taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
            taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(b.data()));
            taskDataSeq->inputs_count.emplace_back(matrix.size());
            taskDataSeq->inputs_count.emplace_back(b.size());
            taskDataSeq->inputs_count.emplace_back(columns);
            taskDataSeq->inputs_count.emplace_back(rows);
            taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expres_seq.data()));
            taskDataSeq->outputs_count.emplace_back(expres_seq.size());

            TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
            ASSERT_EQ(testMpiTaskSequential.validation(), true);
            testMpiTaskSequential.pre_processing();
            testMpiTaskSequential.run();
            testMpiTaskSequential.post_processing();

            ASSERT_EQ(expres_seq, expres_par);

            double eps = 1e-9;
            for (int i = 0; i < rows; i++) {
                double sum = 0;
                for (int j = 0; j < columns; j++) {
                    sum += matrix[i * columns + j] * expres_par[j];
                }
                ASSERT_LT(std::abs(sum - b[i]), eps);
            }
        }
    }

}  // namespace nasedkin_e_seidels_iterate_methods_mpi