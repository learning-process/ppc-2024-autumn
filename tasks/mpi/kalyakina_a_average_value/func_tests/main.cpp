// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kalyakina_a_average_value/include/ops_mpi.hpp"

std::vector<int> RandomVectorWithFixSum(int sum, const int& count) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> result_vector(count);
  for (int i = 0; i < count - 1; i++) {
    result_vector[i] = gen() % (std::min(sum, 255) - 1);
    sum -= result_vector[i];
  }
  result_vector[count - 1] = sum;
  return result_vector;
}

TEST(kalyakina_a_average_value_mpi, Test_Avg_10) {
  boost::mpi::communicator world;
  std::vector<int> in{};
  std::vector<double> out_mpi(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 10;
    const int sum = 1000;
    in = RandomVectorWithFixSum(sum, count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
  }

  kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel AvgMPITaskParallel(taskDataPar);
  ASSERT_EQ(AvgMPITaskParallel.validation(), true);
  AvgMPITaskParallel.pre_processing();
  AvgMPITaskParallel.run();
  AvgMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out_seq(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    // Create Task
    kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential AvgMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(AvgMpiTaskSequential.validation(), true);
    AvgMpiTaskSequential.pre_processing();
    AvgMpiTaskSequential.run();
    AvgMpiTaskSequential.post_processing();

    ASSERT_DOUBLE_EQ(out_mpi[0], out_seq[0]);
  }
}

TEST(kalyakina_a_average_value_mpi, Test_Avg_20) {
  boost::mpi::communicator world;
  std::vector<int> in{};
  std::vector<double> out_mpi(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 20;
    const int sum = 3500;
    in = RandomVectorWithFixSum(sum, count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
  }

  kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel AvgMPITaskParallel(taskDataPar);
  ASSERT_EQ(AvgMPITaskParallel.validation(), true);
  AvgMPITaskParallel.pre_processing();
  AvgMPITaskParallel.run();
  AvgMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out_seq(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    // Create Task
    kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential AvgMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(AvgMpiTaskSequential.validation(), true);
    AvgMpiTaskSequential.pre_processing();
    AvgMpiTaskSequential.run();
    AvgMpiTaskSequential.post_processing();

    ASSERT_DOUBLE_EQ(out_mpi[0], out_seq[0]);
  }
}

TEST(kalyakina_a_average_value_mpi, Test_Avg_50) {
  boost::mpi::communicator world;
  std::vector<int> in{};
  std::vector<double> out_mpi(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 50;
    const int sum = 8000;
    in = RandomVectorWithFixSum(sum, count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
  }

  kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel AvgMPITaskParallel(taskDataPar);
  ASSERT_EQ(AvgMPITaskParallel.validation(), true);
  AvgMPITaskParallel.pre_processing();
  AvgMPITaskParallel.run();
  AvgMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out_seq(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    // Create Task
    kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential AvgMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(AvgMpiTaskSequential.validation(), true);
    AvgMpiTaskSequential.pre_processing();
    AvgMpiTaskSequential.run();
    AvgMpiTaskSequential.post_processing();

    ASSERT_DOUBLE_EQ(out_mpi[0], out_seq[0]);
  }
}

TEST(kalyakina_a_average_value_mpi, Test_Avg_70) {
  boost::mpi::communicator world;
  std::vector<int> in{};
  std::vector<double> out_mpi(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 70;
    const int sum = 10000;
    in = RandomVectorWithFixSum(sum, count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
  }

  kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel AvgMPITaskParallel(taskDataPar);
  ASSERT_EQ(AvgMPITaskParallel.validation(), true);
  AvgMPITaskParallel.pre_processing();
  AvgMPITaskParallel.run();
  AvgMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out_seq(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    // Create Task
    kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential AvgMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(AvgMpiTaskSequential.validation(), true);
    AvgMpiTaskSequential.pre_processing();
    AvgMpiTaskSequential.run();
    AvgMpiTaskSequential.post_processing();

    ASSERT_DOUBLE_EQ(out_mpi[0], out_seq[0]);
  }
}

TEST(kalyakina_a_average_value_mpi, Test_Avg_100) {
  boost::mpi::communicator world;
  std::vector<int> in{};
  std::vector<double> out_mpi(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 100;
    const int sum = 20000;
    in = RandomVectorWithFixSum(sum, count);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_mpi.data()));
    taskDataPar->outputs_count.emplace_back(out_mpi.size());
  }

  kalyakina_a_average_value_mpi::FindingAverageMPITaskParallel AvgMPITaskParallel(taskDataPar);
  ASSERT_EQ(AvgMPITaskParallel.validation(), true);
  AvgMPITaskParallel.pre_processing();
  AvgMPITaskParallel.run();
  AvgMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out_seq(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    // Create Task
    kalyakina_a_average_value_mpi::FindingAverageMPITaskSequential AvgMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(AvgMpiTaskSequential.validation(), true);
    AvgMpiTaskSequential.pre_processing();
    AvgMpiTaskSequential.run();
    AvgMpiTaskSequential.post_processing();

    ASSERT_DOUBLE_EQ(out_mpi[0], out_seq[0]);
  }
}
