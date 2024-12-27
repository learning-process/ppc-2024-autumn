// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/prokhorov_n_global_search_algorithm_strongin/include/ops_mpi.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_strongin_algorithm_quadratic_function) {
  boost::mpi::communicator world;
  std::vector<double> global_a;
  std::vector<double> global_b;
  std::vector<double> global_epsilon;
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_a = {-10.0};
    global_b = {10.0};
    global_epsilon = {0.001};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    taskDataPar->inputs_count.emplace_back(global_a.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    taskDataPar->inputs_count.emplace_back(global_b.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    taskDataPar->inputs_count.emplace_back(global_epsilon.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    taskDataSeq->inputs_count.emplace_back(global_a.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    taskDataSeq->inputs_count.emplace_back(global_b.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    taskDataSeq->inputs_count.emplace_back(global_epsilon.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_NEAR(reference_result[0], global_result[0], 0.001);
  }
}

TEST(prokhorov_n_global_search_algorithm_strongin_mpi, test_strongin_algorithm_sinus_function) {
  boost::mpi::communicator world;
  std::vector<double> global_a;
  std::vector<double> global_b;
  std::vector<double> global_epsilon;
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_a = {0.0};
    global_b = {3.14};
    global_epsilon = {0.001};

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    taskDataPar->inputs_count.emplace_back(global_a.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    taskDataPar->inputs_count.emplace_back(global_b.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    taskDataPar->inputs_count.emplace_back(global_epsilon.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_a.data()));
    taskDataSeq->inputs_count.emplace_back(global_a.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_b.data()));
    taskDataSeq->inputs_count.emplace_back(global_b.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_epsilon.data()));
    taskDataSeq->inputs_count.emplace_back(global_epsilon.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    prokhorov_n_global_search_algorithm_strongin_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_NEAR(reference_result[0], global_result[0], 0.001);
  }
}