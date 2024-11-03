#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>
#include <vector>

#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, test_large_random_points) {
    boost::mpi::communicator world;
    std::vector<double> global_results(1, 0.0);
    int num_points = 1000000;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
        taskDataPar->inputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
        taskDataPar->outputs_count.emplace_back(global_results.size());
    }

    malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
        std::vector<double> reference_results(1, 0.0);
        std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

        taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
        taskDataSeq->inputs_count.emplace_back(1);
        taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_results.data()));
        taskDataSeq->outputs_count.emplace_back(reference_results.size());

        malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential testMpiTaskSequential(taskDataSeq);
        ASSERT_EQ(testMpiTaskSequential.validation(), true);
        testMpiTaskSequential.pre_processing();
        testMpiTaskSequential.run();
        testMpiTaskSequential.post_processing();

        ASSERT_NEAR(reference_results[0], global_results[0], 1e-3);
    }
}

TEST(malyshev_v_monte_carlo_integration_mpi, test_zero_points) {
    boost::mpi::communicator world;
    std::vector<double> global_results(1, 0.0);
    int num_points = 0;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
        taskDataPar->inputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
        taskDataPar->outputs_count.emplace_back(global_results.size());
    }

    malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
        ASSERT_EQ(global_results[0], 0.0);
    }
}

TEST(malyshev_v_monte_carlo_integration_mpi, test_different_ranges) {
    boost::mpi::communicator world;
    std::vector<double> global_results(1, 0.0);
    int num_points = 1000;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
        taskDataPar->inputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
        taskDataPar->outputs_count.emplace_back(global_results.size());
    }

    malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
        std::vector<double> reference_results(1, 0.0);
        std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

        taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
        taskDataSeq->inputs_count.emplace_back(1);
        taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_results.data()));
        taskDataSeq->outputs_count.emplace_back(reference_results.size());

        malyshev_v_monte_carlo_integration::MonteCarloIntegrationSequential testMpiTaskSequential(taskDataSeq);
        ASSERT_EQ(testMpiTaskSequential.validation(), true);
        testMpiTaskSequential.pre_processing();
        testMpiTaskSequential.run();
        testMpiTaskSequential.post_processing();

        ASSERT_NEAR(reference_results[0], global_results[0], 1e-3);
    }
}

TEST(malyshev_v_monte_carlo_integration_mpi, test_edge_case_single_point) {
    boost::mpi::communicator world;
    std::vector<double> global_results(1, 0.0);
    int num_points = 1;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_points));
        taskDataPar->inputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_results.data()));
        taskDataPar->outputs_count.emplace_back(global_results.size());
    }

    malyshev_v_monte_carlo_integration::MonteCarloIntegrationParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
        ASSERT_GT(global_results[0], 0.0);
    }
}
