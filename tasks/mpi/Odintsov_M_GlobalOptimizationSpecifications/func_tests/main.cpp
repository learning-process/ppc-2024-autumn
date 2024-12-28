
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <vector>

#include "mpi/Odintsov_M_GlobalOptimizationSpecifications/include/ops_mpi.hpp"

static std::vector<double> createFunc(int min, int max) {
  std::vector<double> func(2, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min, max);

  for (int i = 0; i < 2; i++) {
    func[i] = dist(gen);
  }

  return func;
}

static std::vector<double> creareConstr(int min, int max, int count) {
  std::vector<double> constr(3 * count, 0);
  srand(time(nullptr));
  for (int i = 0; i < 3 * count; i++) {
    constr[i] = (min + rand() % (max - min + 1));
  }
  return constr;
}
TEST(Odintsov_M_OptimPar_MPI, test_min_0) {
  // Create data
  boost::mpi::communicator com;
  double step = 0.3;
  std::vector<double> area = {0.0000001, 0.0000002, 0.0000001, 0.0000002};
  std::vector<double> func = createFunc(-10, 10);
  std::vector<double> constraint = creareConstr(-10, 10, 1);
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel testClassPar(
      taskDataPar);

  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataSeq->inputs_count.emplace_back(1);  // Количество ограничений
    taskDataSeq->inputs_count.emplace_back(0);  // Режим

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential testClassSeq(
        taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out_s, out);
  }
}
TEST(Odintsov_M_OptimPar_MPI, test_min_1) {
  // Create data
  double step = 0.3;
  boost::mpi::communicator com;
  std::vector<double> area = {-10, 10, -10, 10};
  std::vector<double> func = createFunc(-10, 10);
  std::vector<double> constraint = creareConstr(-10, 10, 36);
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(36);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel testClassPar(
      taskDataPar);

  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataSeq->inputs_count.emplace_back(36);  // Количество ограничений
    taskDataSeq->inputs_count.emplace_back(0);   // Режим

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential testClassSeq(
        taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out_s, out);
  }
}
TEST(Odintsov_M_OptimPar_MPI, test_min_2) {
  // Create data
  boost::mpi::communicator com;
  double step = 0.3;
  std::vector<double> area = {-17, 6, 13, 23};
  std::vector<double> func = createFunc(-10, 10);
  std::vector<double> constraint = creareConstr(-10, 10, 24);
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(24);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel testClassPar(
      taskDataPar);

  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataSeq->inputs_count.emplace_back(24);  // Количество ограничений
    taskDataSeq->inputs_count.emplace_back(0);   // Режим

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential testClassSeq(
        taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out_s, out);
  }
}
TEST(Odintsov_M_OptimPar_MPI, test_max_1) {
  // Create data
  boost::mpi::communicator com;
  double step = 0.3;
  std::vector<double> area = {-20, -10, -20, -10};
  std::vector<double> func = createFunc(-10, 10);
  std::vector<double> constraint = creareConstr(1, 3, 1);
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel testClassPar(
      taskDataPar);

  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataSeq->inputs_count.emplace_back(1);  // Количество ограничений
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential testClassSeq(
        taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out, out_s);
  }
}
TEST(Odintsov_M_OptimPar_MPI, test_min_3) {
  // Create data
  double step = 0.3;
  boost::mpi::communicator com;
  std::vector<double> area = {30, 40, 30, 40};
  std::vector<double> func = createFunc(-10, 10);
  std::vector<double> constraint = creareConstr(-10, -1, 36);
  std::vector<double> out = {0};
  std::vector<double> out_s = {0};
  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataPar->inputs_count.emplace_back(36);
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel testClassPar(
      taskDataPar);

  ASSERT_EQ(testClassPar.validation(), true);
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();
  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(area.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(func.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(constraint.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&step));
    taskDataSeq->inputs_count.emplace_back(36);  // Количество ограничений
    taskDataSeq->inputs_count.emplace_back(0);   // Режим

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(out_s.size());
    Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential testClassSeq(
        taskDataSeq);
    ASSERT_EQ(testClassSeq.validation(), true);
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    ASSERT_EQ(out_s, out);
  }
}