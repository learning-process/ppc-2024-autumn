#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/sotskov_a_sum_element_matrix/include/ops_mpi.hpp"

TEST(sotskov_a_sum_element_matrix_parallel, check_wrong_validation) {
    boost::mpi::communicator world;
    std::vector<int> global_matrix;
    std::vector<int32_t> global_sum(2, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        const int rows = 10;
        const int columns = 15;
        global_matrix = sotskov_a_sum_element_matrix_mpi::create_random_matrix<int>(rows, columns);

        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
        taskDataPar->inputs_count.emplace_back(global_matrix.size());
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());

        sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel<int> testMPITaskParallel(taskDataPar);
        ASSERT_EQ(testMPITaskParallel.validation(), false);  // Expecting false due to incorrect output count
    }
}

TEST(sotskov_a_sum_element_matrix_parallel, check_int_sum) {
    boost::mpi::communicator world;
    std::vector<int> global_matrix;
    std::vector<int32_t> global_sum(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    const int rows = 10;
    const int columns = 15;

    if (world.rank() == 0) {
        global_matrix = sotskov_a_sum_element_matrix_mpi::create_random_matrix<int>(rows, columns);
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
        taskDataPar->inputs_count.emplace_back(global_matrix.size());
        taskDataPar->inputs_count.emplace_back(rows);
        taskDataPar->inputs_count.emplace_back(columns);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel<int> testMPITaskParallel(taskDataPar);
    ASSERT_EQ(testMPITaskParallel.validation(), true);
    testMPITaskParallel.pre_processing();
    testMPITaskParallel.run();
    testMPITaskParallel.post_processing();

    if (world.rank() == 0) {
        std::vector<int32_t> reference_sum(1, 0);
        for (int val : global_matrix) {
            reference_sum[0] += val;
        }
        ASSERT_EQ(reference_sum[0], global_sum[0]);
    }
}

TEST(sotskov_a_sum_element_matrix_parallel, check_double_sum) {
    boost::mpi::communicator world;
    std::vector<double> global_matrix;
    std::vector<double> global_sum(1, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    const int rows = 10;
    const int columns = 15;

    if (world.rank() == 0) {
        global_matrix = sotskov_a_sum_element_matrix_mpi::create_random_matrix<double>(rows, columns);

        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
        taskDataPar->inputs_count.emplace_back(global_matrix.size());
        taskDataPar->inputs_count.emplace_back(rows);
        taskDataPar->inputs_count.emplace_back(columns);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel<double> testMPITaskParallel(taskDataPar);
    ASSERT_EQ(testMPITaskParallel.validation(), true);
    testMPITaskParallel.pre_processing();
    testMPITaskParallel.run();
    testMPITaskParallel.post_processing();

    if (world.rank() == 0) {
        std::vector<double> reference_sum(1, 0.0);
        for (double val : global_matrix) {
            reference_sum[0] += val;
        }
        ASSERT_NEAR(reference_sum[0], global_sum[0], 1e-6);
    }
}

TEST(sotskov_a_sum_element_matrix_parallel, check_empty_matrix) {
    boost::mpi::communicator world;
    std::vector<int> global_matrix;
    std::vector<int32_t> global_sum(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    const int rows = 0;
    const int columns = 0;
    
    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
        taskDataPar->inputs_count.emplace_back(global_matrix.size());
        taskDataPar->inputs_count.emplace_back(rows);
        taskDataPar->inputs_count.emplace_back(columns);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel<int> testMPITaskParallel(taskDataPar);
    ASSERT_EQ(testMPITaskParallel.validation(), true);
    testMPITaskParallel.pre_processing();
    testMPITaskParallel.run();
    testMPITaskParallel.post_processing();

    if (world.rank() == 0) {
        ASSERT_EQ(global_sum[0], 0);
    }
}

TEST(sotskov_a_sum_element_matrix_parallel, check_zero_size) {
    auto zero_columns = sotskov_a_sum_element_matrix_mpi::create_random_matrix<int>(1, 0);
    EXPECT_TRUE(zero_columns.empty());
    auto zero_rows = sotskov_a_sum_element_matrix_mpi::create_random_matrix<int>(0, 1);
    EXPECT_TRUE(zero_rows.empty());
}

TEST(sotskov_a_sum_element_matrix_parallel, check_large_matrix) {
    boost::mpi::communicator world;
    std::vector<int> global_matrix;
    std::vector<int32_t> global_sum(1, 0);
    const int rows = 100;
    const int columns = 100;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        global_matrix = sotskov_a_sum_element_matrix_mpi::create_random_matrix<int>(rows, columns);

        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
        taskDataPar->inputs_count.emplace_back(global_matrix.size());
        taskDataPar->inputs_count.emplace_back(rows);
        taskDataPar->inputs_count.emplace_back(columns);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
        taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel<int> testMPITaskParallel(taskDataPar);
    ASSERT_EQ(testMPITaskParallel.validation(), true);
    testMPITaskParallel.pre_processing();
    testMPITaskParallel.run();
    testMPITaskParallel.post_processing();

    if (world.rank() == 0) {
        std::vector<int32_t> reference_sum(1, 0);
        for (int val : global_matrix) {
            reference_sum[0] += val;
        }
        ASSERT_EQ(reference_sum[0], global_sum[0]);
    }
}
