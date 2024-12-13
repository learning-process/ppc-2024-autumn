#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include "../include/ops_mpi.hpp"

TEST(Strassen_Algorithm_MPI, Test_Matrix_2x2) {
    boost::mpi::communicator world;
    std::vector<int> A, B;
    std::vector<int> C(2 * 2, 0);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        A = nasedkin_e_strassen_algorithm::getRandomMatrix(2);
        B = nasedkin_e_strassen_algorithm::getRandomMatrix(2);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
        taskData->inputs_count.emplace_back(A.size());
        taskData->inputs_count.emplace_back(B.size());
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
        taskData->outputs_count.emplace_back(C.size());
    }

    nasedkin_e_strassen_algorithm::StrassenMPITaskParallel strassenTask(taskData);
    ASSERT_EQ(strassenTask.validation(), true);
    strassenTask.pre_processing();
    strassenTask.run();
    strassenTask.post_processing();

    if (world.rank() == 0) {
        std::vector<int> reference_C = nasedkin_e_strassen_algorithm::matrixMultiply(A, B, 2);
        ASSERT_EQ(C, reference_C);
    }
}

TEST(Strassen_Algorithm_MPI, Test_Matrix_4x4) {
    boost::mpi::communicator world;
    std::vector<int> A, B;
    std::vector<int> C(4 * 4, 0);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        A = nasedkin_e_strassen_algorithm::getRandomMatrix(4);
        B = nasedkin_e_strassen_algorithm::getRandomMatrix(4);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
        taskData->inputs_count.emplace_back(A.size());
        taskData->inputs_count.emplace_back(B.size());
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
        taskData->outputs_count.emplace_back(C.size());
    }

    nasedkin_e_strassen_algorithm::StrassenMPITaskParallel strassenTask(taskData);
    ASSERT_EQ(strassenTask.validation(), true);
    strassenTask.pre_processing();
    strassenTask.run();
    strassenTask.post_processing();

    if (world.rank() == 0) {
        std::vector<int> reference_C = nasedkin_e_strassen_algorithm::matrixMultiply(A, B, 4);
        ASSERT_EQ(C, reference_C);
    }
}

TEST(Strassen_Algorithm_MPI, Test_Matrix_8x8) {
    boost::mpi::communicator world;
    std::vector<int> A, B;
    std::vector<int> C(8 * 8, 0);
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        A = nasedkin_e_strassen_algorithm::getRandomMatrix(8);
        B = nasedkin_e_strassen_algorithm::getRandomMatrix(8);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
        taskData->inputs_count.emplace_back(A.size());
        taskData->inputs_count.emplace_back(B.size());
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
        taskData->outputs_count.emplace_back(C.size());
    }

    nasedkin_e_strassen_algorithm::StrassenMPITaskParallel strassenTask(taskData);
    ASSERT_EQ(strassenTask.validation(), true);
    strassenTask.pre_processing();
    strassenTask.run();
    strassenTask.post_processing();

    if (world.rank() == 0) {
        std::vector<int> reference_C = nasedkin_e_strassen_algorithm::matrixMultiply(A, B, 8);
        ASSERT_EQ(C, reference_C);
    }
}