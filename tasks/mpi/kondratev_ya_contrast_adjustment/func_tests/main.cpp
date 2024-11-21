// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kondratev_ya_contrast_adjustment/include/ops_mpi.hpp"

namespace kondratev_ya_contrast_adjustment_mpi {
std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> genGradient(uint32_t side) {
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> buff(side * side);
  auto step = (uint8_t)(255 / (2 * side - 1));

  for (uint32_t i = 0; i < side; i++) {
    for (uint32_t j = i; j < side; j++) {
      auto ind = i * side + j;
      auto ind2 = j * side + i;

      buff[ind] = step * (i + j + 1);
      buff[ind2] = step * (i + j + 1);
    }
  }
  return buff;
}

std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> genRandomData(uint32_t size) {
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> buff(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (uint32_t i = 0; i < size; i++) {
    buff[i] = gen() % 256;
  }
  return buff;
}
}  // namespace kondratev_ya_contrast_adjustment_mpi

TEST(kondratev_ya_contrast_adjustment_mpi, gradient_test_increase) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int side = 24;
    input = kondratev_ya_contrast_adjustment_mpi::genGradient(side);
    res.resize(input.size());
    contrast = std::make_shared<double>(1.25);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> ref;
    ref.resize(input.size());

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref.data()));
    taskDataSeq->outputs_count.emplace_back(ref.size());

    kondratev_ya_contrast_adjustment_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    auto inputContrast = kondratev_ya_contrast_adjustment_mpi::getContrast(input);
    auto resContrast = kondratev_ya_contrast_adjustment_mpi::getContrast(res);
    auto refContrast = kondratev_ya_contrast_adjustment_mpi::getContrast(ref);

    for (uint32_t i = 0; i < ref.size(); i++) {
      ASSERT_EQ(ref[i].red, res[i].red);
      ASSERT_EQ(ref[i].green, res[i].green);
      ASSERT_EQ(ref[i].blue, res[i].blue);
    }

    ASSERT_GE(refContrast, inputContrast);
    ASSERT_DOUBLE_EQ(refContrast, resContrast);
  }
}

TEST(kondratev_ya_contrast_adjustment_mpi, gradient_test_decrease) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int side = 24;
    input = kondratev_ya_contrast_adjustment_mpi::genGradient(side);
    res.resize(input.size());
    contrast = std::make_shared<double>(0.25);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> ref;
    ref.resize(input.size());

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref.data()));
    taskDataSeq->outputs_count.emplace_back(ref.size());

    kondratev_ya_contrast_adjustment_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    auto inputContrast = kondratev_ya_contrast_adjustment_mpi::getContrast(input);
    auto resContrast = kondratev_ya_contrast_adjustment_mpi::getContrast(res);
    auto refContrast = kondratev_ya_contrast_adjustment_mpi::getContrast(ref);

    for (uint32_t i = 0; i < ref.size(); i++) {
      ASSERT_EQ(ref[i].red, res[i].red);
      ASSERT_EQ(ref[i].green, res[i].green);
      ASSERT_EQ(ref[i].blue, res[i].blue);
    }

    ASSERT_LE(refContrast, inputContrast);
    ASSERT_DOUBLE_EQ(refContrast, resContrast);
  }
}

TEST(kondratev_ya_contrast_adjustment_mpi, random_test_increase) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int side = 24 * 24;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(side);
    res.resize(input.size());
    contrast = std::make_shared<double>(1.25);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> ref;
    ref.resize(input.size());

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref.data()));
    taskDataSeq->outputs_count.emplace_back(ref.size());

    kondratev_ya_contrast_adjustment_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (uint32_t i = 0; i < ref.size(); i++) {
      ASSERT_EQ(ref[i].red, res[i].red);
      ASSERT_EQ(ref[i].green, res[i].green);
      ASSERT_EQ(ref[i].blue, res[i].blue);
    }
  }
}

TEST(kondratev_ya_contrast_adjustment_mpi, random_test_decrease) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int side = 24 * 24;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(side);
    res.resize(input.size());
    contrast = std::make_shared<double>(0.25);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> ref;
    ref.resize(input.size());

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref.data()));
    taskDataSeq->outputs_count.emplace_back(ref.size());

    kondratev_ya_contrast_adjustment_mpi::TestMPITaskSequential testTaskSequential(taskDataSeq);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    for (uint32_t i = 0; i < ref.size(); i++) {
      ASSERT_EQ(ref[i].red, res[i].red);
      ASSERT_EQ(ref[i].green, res[i].green);
      ASSERT_EQ(ref[i].blue, res[i].blue);
    }
  }
}
