// Copyright 2024 Tarakanov Denis
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>

#include "mpi/tarakanov_d_integration_the_trapezoid_method/include/ops_mpi.hpp"

TEST(tarakanov_d_integration_the_trapezoid_method_mpi_func_tests, Test_Integration1) {
  boost::mpi::communicator world;
  std::vector<double> global_res(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 1.0;
  double h = 1e-8;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par parallelTask(taskDataPar);

  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(1, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
    taskDataSeq->inputs_count.emplace_back(3);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_seq sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_res[0], global_res[0], 0.1);
  }
}

TEST(tarakanov_d_integration_the_trapezoid_method_mpi_func_tests, Test_Integration2) {
  boost::mpi::communicator world;
  std::vector<double> global_res(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 5.0;
  double b = 7.0;
  double h = 1e-8;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par parallelTask(taskDataPar);

  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(1, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
    taskDataSeq->inputs_count.emplace_back(3);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_seq sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_res[0], global_res[0], 0.1);
  }
}

TEST(tarakanov_d_integration_the_trapezoid_method_mpi_func_tests, Test_Integration3) {
  boost::mpi::communicator world;
  std::vector<double> global_res(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = -2.0;
  double b = -1.0;
  double h = 1e-8;

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
    taskDataPar->inputs_count.emplace_back(3);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_par parallelTask(taskDataPar);

  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(1, 0.0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&h));
    taskDataSeq->inputs_count.emplace_back(3);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(1);

    tarakanov_d_integration_the_trapezoid_method_mpi::integration_the_trapezoid_method_seq sequentialTask(taskDataSeq);
    ASSERT_EQ(sequentialTask.validation(), true);
    sequentialTask.pre_processing();
    sequentialTask.run();
    sequentialTask.post_processing();

    ASSERT_NEAR(reference_res[0], global_res[0], 0.1);
  }
}