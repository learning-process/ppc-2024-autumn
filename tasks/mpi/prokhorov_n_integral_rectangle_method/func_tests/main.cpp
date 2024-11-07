#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/prokhorov_n_integral_rectangle_method/include/ops_mpi.hpp"

TEST(prokhorov_n_integral_rectangle_method_mpi, Test_Sine) {
  boost::mpi::communicator world;
  std::vector<double> global_input(3);
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_input[0] = 0.0;
    global_input[1] = M_PI;
    global_input[2] = 1000;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function([](double x) { return std::sin(x); });
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataSeq->inputs_count.emplace_back(global_input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    taskDataSeq->outputs_count.emplace_back(reference_result.size());

    prokhorov_n_integral_rectangle_method_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.set_function([](double x) { return std::sin(x); });
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(reference_result[0], global_result[0], 1e-5);
  }
}
TEST(prokhorov_n_integral_rectangle_method_mpi, Test_Cosine) {
  boost::mpi::communicator world;
  std::vector<double> global_input(3);
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_input[0] = 0.0;
    global_input[1] = M_PI / 2;
    global_input[2] = 1000;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function([](double x) { return std::cos(x); });
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(global_result[0], 1.0, 1e-5);
  }
}

TEST(prokhorov_n_integral_rectangle_method_mpi, Test_Logarithm) {
  boost::mpi::communicator world;
  std::vector<double> global_input(3);
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_input[0] = 0.0;
    global_input[1] = 1.0;
    global_input[2] = 1000;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function([](double x) { return std::log(x + 1); });
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(global_result[0], 0.386294, 1e-5);
  }
}

TEST(prokhorov_n_integral_rectangle_method_mpi, Test_Cube) {
  boost::mpi::communicator world;
  std::vector<double> global_input(3);
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_input[0] = 0.0;
    global_input[1] = 1.0;
    global_input[2] = 1000;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function([](double x) { return x * x * x; });
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(global_result[0], 0.25, 1e-5);
  }
}

TEST(prokhorov_n_integral_rectangle_method_mpi, Test_Reciprocal_Square_Plus_One) {
  boost::mpi::communicator world;
  std::vector<double> global_input(3);
  std::vector<double> global_result(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_input[0] = 0.0;
    global_input[1] = 1.0;
    global_input[2] = 1000;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  prokhorov_n_integral_rectangle_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function([](double x) { return 1.0 / (1.0 + x * x); });
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_NEAR(global_result[0], M_PI / 4, 1e-5);
  }
}
