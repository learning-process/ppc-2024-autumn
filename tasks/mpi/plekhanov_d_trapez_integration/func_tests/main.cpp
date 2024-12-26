#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/plekhanov_d_trapez_integration/include/ops_mpi.hpp"

auto func1 = [](double x) { return x * x + 2 * x + 1; };
auto func2 = [](double x) { return std::pow(x, 3) + 2 * x * x + 8; };
auto func3 = [](double x) { return std::cos(x); };
auto func4 = [](double x) { return std::sin(x); };
auto func5 = [](double x) { return std::pow(3, x); };
auto func6 = [](double x) { return std::exp(x * 2); };
auto func7 = [](double x) { return std::pow(x, 4) - std::exp(x) + std::pow(4, x); };

namespace plekhanov_d_trapez_integration_mpi {

void run_test(double a_, double b_, const std::function<double(double)>& f_) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = a_;
  double b = b_;
  int n = 100000;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  plekhanov_d_trapez_integration_mpi::trapezIntegrationMPI testMpiTaskParallel(taskDataPar);

  testMpiTaskParallel.set_function(f_);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    plekhanov_d_trapez_integration_mpi::trapezIntegrationSEQ testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.set_function(f_);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(global_result[0], reference_result[0], 0.01);
  }
}

}  // namespace plekhanov_d_trapez_integration_mpi

TEST(plekhanov_d_trapez_integration_mpi, test_int_squared_func) {
  plekhanov_d_trapez_integration_mpi::run_test(-1, 4, func1);
}

TEST(plekhanov_d_trapez_integration_mpi, test_int_trippled_func) {
  plekhanov_d_trapez_integration_mpi::run_test(0, 10, func2);
}

TEST(plekhanov_d_trapez_integration_mpi, test_int_cosine_func) {
  plekhanov_d_trapez_integration_mpi::run_test(M_PI, M_PI * 2, func3);
}

TEST(plekhanov_d_trapez_integration_mpi, test_int_sine_func) {
  plekhanov_d_trapez_integration_mpi::run_test(M_PI, M_PI * 4, func4);
}

TEST(plekhanov_d_trapez_integration_mpi, test_int_pow_func) {
  plekhanov_d_trapez_integration_mpi::run_test(-1, 6, func5);
}

TEST(plekhanov_d_trapez_integration_mpi, test_int_exp_func) {
  plekhanov_d_trapez_integration_mpi::run_test(-2.0, 8.0, func6);
}

TEST(plekhanov_d_trapez_integration_mpi, test_int_mixed_func) {
  plekhanov_d_trapez_integration_mpi::run_test(-2.0, 10.0, func7);
}
