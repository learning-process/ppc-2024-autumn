#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <vector>

#include "mpi/prokhorov_n_rectangular_integration/include/ops_mpi.hpp"

TEST(prokhorov_n_rectangular_integration_mpi, test_integration_cos_x) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, M_PI / 2.0, 1000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  taskDataPar->inputs_count.emplace_back(global_input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataPar->outputs_count.emplace_back(global_result.size());

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return std::cos(x); });

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataSeq->inputs_count.emplace_back(global_input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    auto testMpiTaskSequential =
        std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential>(taskDataSeq);
    testMpiTaskSequential->set_function([](double x) { return std::cos(x); });

    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
  }
}

TEST(prokhorov_n_rectangular_integration_mpi, test_integration_x_cubed) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {0.0, 1.0, 1000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  taskDataPar->inputs_count.emplace_back(global_input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataPar->outputs_count.emplace_back(global_result.size());

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return x * x * x; });

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataSeq->inputs_count.emplace_back(global_input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    auto testMpiTaskSequential =
        std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential>(taskDataSeq);
    testMpiTaskSequential->set_function([](double x) { return x * x * x; });

    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
  }
}

TEST(prokhorov_n_rectangular_integration_mpi, test_integration_one_over_x) {
  boost::mpi::communicator world;
  std::vector<double> global_input = {1.0, 2.0, 1000.0};
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
  taskDataPar->inputs_count.emplace_back(global_input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataPar->outputs_count.emplace_back(global_result.size());

  auto testMpiTaskParallel =
      std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskParallel>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return 1.0 / x; });

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataSeq->inputs_count.emplace_back(global_input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    auto testMpiTaskSequential =
        std::make_shared<prokhorov_n_rectangular_integration_mpi::TestMPITaskSequential>(taskDataSeq);
    testMpiTaskSequential->set_function([](double x) { return 1.0 / x; });

    ASSERT_EQ(testMpiTaskSequential->validation(), true);
    testMpiTaskSequential->pre_processing();
    testMpiTaskSequential->run();
    testMpiTaskSequential->post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-3);
  }
}
