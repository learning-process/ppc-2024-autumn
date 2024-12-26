// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "mpi/veliev_e_sobel_operator/include/ops_mpi.hpp"

namespace veliev_e_sobel_operator_mpi {
std::vector<double> create_random_vector(int size) {
  std::uniform_real_distribution<double> unif(static_cast<double>(0), static_cast<double>(255));
  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());
  std::vector<double> tmp;
  tmp.reserve(size);
  std::generate_n(std::back_inserter(tmp), size, [&]() { return unif(rand_engine); });

  return tmp;
}
}  // namespace veliev_e_sobel_operator_mpi

TEST(veliev_e_sobel_operator_mpi, TestStart) {
  boost::mpi::communicator world;
  int h = 5;
  int w = 5;
  std::vector<double> in = {220, 220, 220, 50,  50,  220, 220, 220, 50,  50,  220, 220, 220,
                            50,  50,  220, 220, 220, 50,  50,  220, 220, 220, 50,  50};
  std::vector<double> out(in.size(), 0);
  std::vector<double> outex = {0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  veliev_e_sobel_operator_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(h);
    taskDataSeq->inputs_count.emplace_back(w);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    veliev_e_sobel_operator_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    /*std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        std::cout << std::setw(8) << in[i * w + j];
      }
      std::cout << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        std::cout << std::setw(8) << out[i * w + j];
      }
      std::cout << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        std::cout << std::setw(8) << out1[i * w + j];
      }
      std::cout << std::endl;
    }*/

    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
      ASSERT_NEAR(out[i], outex[i], 1e-10);
    }
  }
}

TEST(veliev_e_sobel_operator_mpi, TestStart1) {
  boost::mpi::communicator world;
  int h = 5;
  int w = 5;
  std::vector<double> in = veliev_e_sobel_operator_mpi::create_random_vector(h * w);
  std::vector<double> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  veliev_e_sobel_operator_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(h);
    taskDataSeq->inputs_count.emplace_back(w);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    veliev_e_sobel_operator_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
    }
  }
}

TEST(veliev_e_sobel_operator_mpi, TestStart2) {
  boost::mpi::communicator world;
  int h = 200;
  int w = 200;
  std::vector<double> in = veliev_e_sobel_operator_mpi::create_random_vector(h * w);
  std::vector<double> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  veliev_e_sobel_operator_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(h);
    taskDataSeq->inputs_count.emplace_back(w);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    veliev_e_sobel_operator_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
    }
  }
}

TEST(veliev_e_sobel_operator_mpi, TestStart3) {
  boost::mpi::communicator world;
  int h = 30;
  int w = 70;
  std::vector<double> in = veliev_e_sobel_operator_mpi::create_random_vector(h * w);
  std::vector<double> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  veliev_e_sobel_operator_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(h);
    taskDataSeq->inputs_count.emplace_back(w);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    veliev_e_sobel_operator_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
    }
  }
}

TEST(veliev_e_sobel_operator_mpi, TestStart4) {
  boost::mpi::communicator world;
  int h = 321;
  int w = 243;
  std::vector<double> in = veliev_e_sobel_operator_mpi::create_random_vector(h * w);
  std::vector<double> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  veliev_e_sobel_operator_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(h);
    taskDataSeq->inputs_count.emplace_back(w);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    veliev_e_sobel_operator_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
    }
  }
}

TEST(veliev_e_sobel_operator_mpi, Testval) {
  boost::mpi::communicator world;
  int h = 1;
  int w = 243;
  std::vector<double> in = veliev_e_sobel_operator_mpi::create_random_vector(h * w);
  std::vector<double> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  veliev_e_sobel_operator_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(h);
    taskDataSeq->inputs_count.emplace_back(w);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    veliev_e_sobel_operator_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}
TEST(veliev_e_sobel_operator_mpi, Testval1) {
  boost::mpi::communicator world;
  int h = 111;
  int w = 1;
  std::vector<double> in = veliev_e_sobel_operator_mpi::create_random_vector(h * w);
  std::vector<double> out(in.size(), 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->inputs_count.emplace_back(h);
    taskDataPar->inputs_count.emplace_back(w);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  veliev_e_sobel_operator_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out1(in.size(), 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->inputs_count.emplace_back(h);
    taskDataSeq->inputs_count.emplace_back(w);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    // Create Task
    veliev_e_sobel_operator_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}
