// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/kondratev_ya_contrast_adjustment/include/ops_mpi.hpp"

namespace kondratev_ya_contrast_adjustment_mpi {
std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> genGradient(uint32_t height, uint32_t width) {
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> buff(height * width);
  auto step = (uint8_t)(255 / (height + width - 1));

  for (uint32_t i = 0; i < height; i++) {
    for (uint32_t j = 0; j < width; j++) {
      buff[i * width + j] = step * (i + j + 1);
    }
  }
  return buff;
}

std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> genGradient(uint32_t side) {
  return kondratev_ya_contrast_adjustment_mpi::genGradient(side, side);
}

std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> genRandomData(uint32_t height, uint32_t width) {
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> buff(height * width);
  std::random_device dev;
  std::mt19937 gen(dev());

  for (uint32_t i = 0; i < height; i++) {
    for (uint32_t j = 0; j < width; j++) {
      buff[i * width + j] = gen() % 256;
    }
  }
  return buff;
}

std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> genRandomData(uint32_t side) {
  return kondratev_ya_contrast_adjustment_mpi::genRandomData(side, side);
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

TEST(kondratev_ya_contrast_adjustment_mpi, gradient_test_increase_with_non_equal_sizes) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int height = 8;
    int width = 17;
    input = kondratev_ya_contrast_adjustment_mpi::genGradient(height, width);
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

TEST(kondratev_ya_contrast_adjustment_mpi, gradient_test_decrease_with_non_equal_sizes) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int height = 32;
    int width = 11;
    input = kondratev_ya_contrast_adjustment_mpi::genGradient(height, width);
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
    int side = 24;
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
    int side = 24;
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

TEST(kondratev_ya_contrast_adjustment_mpi, random_test_increase_with_non_equal_sizes) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int height = 12;
    int width = 15;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(height, width);
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

TEST(kondratev_ya_contrast_adjustment_mpi, random_test_decrease_with_non_equal_sizes) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int height = 27;
    int width = 7;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(height, width);
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

TEST(kondratev_ya_contrast_adjustment_mpi, bad_contrast) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int height = 27;
    int width = 7;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(height, width);
    res.resize(input.size());
    contrast = std::make_shared<double>(-0.25);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  bool ret = testMpiTaskParallel.validation();
  if (world.rank() == 0)
    ASSERT_EQ(ret, false);
  else
    ASSERT_EQ(ret, true);
}

TEST(kondratev_ya_contrast_adjustment_mpi, different_sizes) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int side = 24;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(side);
    contrast = std::make_shared<double>(0.25);

    res.resize(input.size() + 1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  bool ret = testMpiTaskParallel.validation();
  if (world.rank() == 0)
    ASSERT_EQ(ret, false);
  else
    ASSERT_EQ(ret, true);
}

TEST(kondratev_ya_contrast_adjustment_mpi, no_out) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int side = 24;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(side);
    contrast = std::make_shared<double>(0.25);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs_count.emplace_back(input.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  bool ret = testMpiTaskParallel.validation();
  if (world.rank() == 0)
    ASSERT_EQ(ret, false);
  else
    ASSERT_EQ(ret, true);
}

TEST(kondratev_ya_contrast_adjustment_mpi, no_in) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  if (world.rank() == 0) {
    int side = 24;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(side);
    contrast = std::make_shared<double>(0.25);
    res.resize(input.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  bool ret = testMpiTaskParallel.validation();
  if (world.rank() == 0)
    ASSERT_EQ(ret, false);
  else
    ASSERT_EQ(ret, true);
}

TEST(kondratev_ya_contrast_adjustment_mpi, no_constast) {
  boost::mpi::communicator world;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int side = 24;
    input = kondratev_ya_contrast_adjustment_mpi::genRandomData(side);
    res.resize(input.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  kondratev_ya_contrast_adjustment_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  bool ret = testMpiTaskParallel.validation();
  if (world.rank() == 0)
    ASSERT_EQ(ret, false);
  else
    ASSERT_EQ(ret, true);
}
