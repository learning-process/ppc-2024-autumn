#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

TEST(malyshev_v_monte_carlo_integration_mpi, Test_Integration_mpi) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 1.0;
  double epsilon = 0.0004;  // Further relaxed tolerance

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  malyshev_v_monte_carlo_integration::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    double expected_value = 0.3333;  // Expected integral value for ∫(x^2) dx from 0 to 1
    ASSERT_NEAR(global_result[0], expected_value, epsilon);
  }
}
