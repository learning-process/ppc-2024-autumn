#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>

#include "mpi/burykin_m_strongin/include/ops_mpi.hpp"

TEST(burykin_m_strongin_MPI, TestNumber1) {
  boost::mpi::communicator world;
  double x0 = 1.0;
  double x1 = 1000.0;
  double eps = 0.01;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginParallel testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    double ref_res = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(1, res, eps);
  }
}

TEST(burykin_m_strongin_MPI, TestNumber2) {
  boost::mpi::communicator world;
  double x0 = 1000.0;
  double x1 = 2500.0;
  double eps = 0.001;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginParallel testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    double ref_res = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(1000, res, eps);
  }
}

TEST(burykin_m_strongin_MPI, TestNumber3) {
  boost::mpi::communicator world;
  double x0 = 0.0;
  double x1 = 5.0;
  double eps = 0.00000001;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginParallel testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    double ref_res = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(0, res, eps);
  }
}

TEST(burykin_m_strongin_MPI, TestNumber4) {
  boost::mpi::communicator world;
  double x0 = 1000.0;
  double x1 = 1.0;
  double eps = 0.01;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginParallel testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    double ref_res = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(1, res, eps);
  }
}

TEST(burykin_m_strongin_MPI, TestNumber5) {
  boost::mpi::communicator world;
  double x0 = 70.0;
  double x1 = 10000.0;
  double eps = 0.0001;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginParallel testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    double ref_res = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(70, res, eps);
  }
}

TEST(burykin_m_strongin_MPI, TestNumber6) {
  boost::mpi::communicator world;
  double x0 = 567.0;
  double x1 = 10765.0;
  double eps = 0.00001;
  double res = 0;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto fn = [](double x) { return x * x; };
  burykin_m_strongin::StronginParallel testMpiTaskParallel(taskDataPar, fn);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    double ref_res = 0;

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x0));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&x1));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eps));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&ref_res));
    taskDataSeq->outputs_count.emplace_back(1);

    // Create Task
    burykin_m_strongin::StronginSequential testMpiTaskSequential(taskDataSeq, fn);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_NEAR(567, res, eps);
  }
}
