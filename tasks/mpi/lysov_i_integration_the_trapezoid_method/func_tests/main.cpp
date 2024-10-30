#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>

#include "mpi/lysov_i_integration_the_trapezoid_method/include/ops_mpi.hpp"

TEST(lysov_i_integration_the_trapezoid_method_mpi, Test_Integration_mpi_1) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = -1.45;
  double b = 0.0;
  double epsilon = 0.01;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }
  lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(lysov_i_integration_the_trapezoid_method_mpi, Test_Integration_mpi_2) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.45;
  double epsilon = 0.01;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }
  lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(lysov_i_integration_the_trapezoid_method_mpi, Test_Integration_mpi_5) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = -10.0;
  double b = 65.0;
  double epsilon = 0.00001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(lysov_i_integration_the_trapezoid_method_mpi, Test_Integration_mpi_6) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = 0.0;
  double b = 100.0;
  double epsilon = 0.00001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(lysov_i_integration_the_trapezoid_method_mpi, Test_Integration_mpi_7) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  double a = -5.0;
  double b = 5.0;
  double epsilon = 0.000001;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  }

  lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

   if (world.rank() == 0) {
    std::vector<double> reference_result(1, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_result.data()));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_NEAR(reference_result[0], global_result[0], 1e-1);
  }
}

TEST(lysov_i_integration_the_trapezoid_method_mpi, TaskMpi_InputSizeLessThan3) {
  std::shared_ptr<ppc::core::TaskData> taskDataMPIParallel = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double a = -1.0;
    double b = 1.0;
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    double result = 0.0;
    taskDataMPIParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testTaskMPIParallel(taskDataMPIParallel);
    ASSERT_EQ(testTaskMPIParallel.validation(), false);
  }
}

TEST(lysov_i_integration_the_trapezoid_method_mpi, TaskMpi_InputSizeMoreThan3) {
  std::shared_ptr<ppc::core::TaskData> taskDataMPIParallel = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double a = -1.0;
    double b = 1.0;
    double epsilon = 0.01;
    double extra_input = 5.0;
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&extra_input));
    double result = 0.0;
    taskDataMPIParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testTaskMPIParallel(taskDataMPIParallel);
    ASSERT_EQ(testTaskMPIParallel.validation(), false);
  }
}

TEST(lysov_i_integration_the_trapezoid_method_mpi, TaskMpi_OutputSizeMoreThan1) {
  std::shared_ptr<ppc::core::TaskData> taskDataMPIParallel = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double a = -1.0;
    double b = 1.0;
    double epsilon = 0.01;
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    double result1 = 0.0;
    double result2 = 0.0;
    taskDataMPIParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result1));
    taskDataMPIParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result2));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testTaskMPIParallel(taskDataMPIParallel);
    ASSERT_EQ(testTaskMPIParallel.validation(), false);
  }
}

TEST(lysov_i_integration_the_trapezoid_method_mpi, TaskMpi_OutputSizeLessThan1) {
  std::shared_ptr<ppc::core::TaskData> taskDataMPIParallel = std::make_shared<ppc::core::TaskData>();
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    double a = -1.0;
    double b = 1.0;
    double epsilon = 0.01;
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataMPIParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    lysov_i_integration_the_trapezoid_method_mpi::TestMPITaskParallel testTaskMPIParallel(taskDataMPIParallel);
    ASSERT_EQ(testTaskMPIParallel.validation(), false);
  }
}
