#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/smirnov_i_integration_by_rectangles/include/ops_mpi.hpp"
double f1(double x) { return x * x; }
double f2(double x) { return std::exp(x); }
double f3(double x) { return std::sin(x); }
double f_const(double x) { return 5 + 0 * x; }
double f_lin(double x) { return x; }

TEST(smirnov_i_integration_by_rectangles_mpi, Test_const) {
  boost::mpi::communicator world;
  double left = 0.0;
  double right = 1.0;
  int n_ = 1000;
  std::vector<double> global_res(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::uint32_t o_size = 1;
    std::uint32_t i_size = 3;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataPar->inputs_count.emplace_back(i_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(o_size);
  }

  smirnov_i_integration_by_rectangles::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function(f_const);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    testMpiTaskSequential.set_function(f_const);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(result_seq[0], global_res[0], 1e-3);
  }
}

TEST(smirnov_i_integration_by_rectangles_mpi, Test_linear) {
  boost::mpi::communicator world;
  double left = 0.0;
  double right = 1.0;
  int n_ = 1000;
  std::vector<double> global_res(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  smirnov_i_integration_by_rectangles::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function(f_lin);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    testMpiTaskSequential.set_function(f_lin);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(result_seq[0], global_res[0], 1e-3);
  }
}
TEST(smirnov_i_integration_by_rectangles, Test_x_times_x) {
  boost::mpi::communicator world;
  double left = 0.0;
  double right = 1.0;
  int n_ = 1000;
  std::vector<double> global_res(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::uint32_t o_size = 1;
    std::uint32_t i_size = 3;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataPar->inputs_count.emplace_back(i_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(o_size);
  }
  smirnov_i_integration_by_rectangles::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function(f1);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    testMpiTaskSequential.set_function(f1);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(result_seq[0], global_res[0], 1e-3);
  }
}

TEST(smirnov_i_integration_by_rectangles_mpi, Test_e_x) {
  boost::mpi::communicator world;
  double left = 0.0;
  double right = 1.0;
  int n_ = 1000;
  std::vector<double> global_res(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  smirnov_i_integration_by_rectangles::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function(f2);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    testMpiTaskSequential.set_function(f2);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(result_seq[0], global_res[0], 1e-3);
  }
}

TEST(smirnov_i_integration_by_rectangles_mpi, Test_sin_x) {
  boost::mpi::communicator world;
  double left = 0.0;
  double right = 1.0;
  int n_ = 1000;
  std::vector<double> global_res(1, 0.0);
  std::vector<double> result_seq(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  smirnov_i_integration_by_rectangles::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.set_function(f3);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&left));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&right));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n_));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_seq.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    smirnov_i_integration_by_rectangles::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

    testMpiTaskSequential.set_function(f3);

    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(result_seq[0], global_res[0], 1e-3);
  }
}
