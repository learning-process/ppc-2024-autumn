// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mpi/beresnev_a_cannons_algorithm/include/ops_mpi.hpp"

static std::vector<double> getRandomVector(int sz, double d) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-d, d);
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Empty_in) {
  boost::mpi::communicator world;
  int n = 3;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inB = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Empty_out) {
  boost::mpi::communicator world;
  int n = 3;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = getRandomVector(n * n, dist);
    inB = getRandomVector(n * n, dist);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Wrong_Size) {
  boost::mpi::communicator world;
  int n = 3;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = getRandomVector(n * n, dist);
    inB = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size() - 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Wrong_Size_1) {
  boost::mpi::communicator world;
  int n = 3;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = getRandomVector(n * n, dist);
    inB = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size() + 1);
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMPITaskParallel.validation(), false);
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_m1_1) {
  boost::mpi::communicator world;
  int n = 1;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = getRandomVector(n * n, dist);
    inB = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference(n * n);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataSeq->inputs_count.emplace_back(inA.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataSeq->inputs_count.emplace_back(inB.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&reference));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    beresnev_a_cannons_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_TRUE(std::equal(reference.begin(), reference.end(), outC.begin(),
                           [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Inverse) {
  boost::mpi::communicator world;
  int n = 5;
  std::vector<double> iden = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
  std::vector<double> inA = {-0.5, -0.5, 0,    0.5, 0,    0.5, 0,   -0.25, 0, -0.5, 1.5, 0.5, -0.25,
                             -0.5, 0,    -0.5, 0.5, 0.25, 0,   0.5, -0.5,  0, 0.25, 0,   0};
  std::vector<double> inB = {2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 4, 0, 4, 0, 8, 4, 2, 2, 2, 2, 0, -2, 0, 0, -2};
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(
        std::equal(iden.begin(), iden.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Iden) {
  boost::mpi::communicator world;
  int n = 5;
  double dist = 100.0;
  std::vector<double> inA = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inB = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(
        std::equal(inB.begin(), inB.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Iden_1) {
  boost::mpi::communicator world;
  int n = 5;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(
        std::equal(inA.begin(), inA.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Random) {
  boost::mpi::communicator world;
  int n = 7;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = getRandomVector(n * n, dist);
    inB = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference(n * n);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataSeq->inputs_count.emplace_back(inA.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataSeq->inputs_count.emplace_back(inB.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&reference));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    beresnev_a_cannons_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < reference.size(); ++i) {
      if (std::abs(reference[i] - outC[i]) >= 1e-9) {
        std::cerr << "Mismatch at index " << i << ": reference = " << reference[i] << ", outC = " << outC[i]
                  << ", diff = " << std::abs(reference[i] - outC[i]) << std::endl;
      }
    }
    ASSERT_TRUE(std::equal(reference.begin(), reference.end(), outC.begin(),
                           [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Random_1) {
  boost::mpi::communicator world;
  int n = 80;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = getRandomVector(n * n, dist);
    inB = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference(n * n);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataSeq->inputs_count.emplace_back(inA.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataSeq->inputs_count.emplace_back(inB.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&reference));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    beresnev_a_cannons_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < reference.size(); ++i) {
      if (std::abs(reference[i] - outC[i]) >= 1e-9) {
        std::cerr << "Mismatch at index " << i << ": reference = " << reference[i] << ", outC = " << outC[i]
                  << ", diff = " << std::abs(reference[i] - outC[i]) << std::endl;
      }
    }
    ASSERT_TRUE(std::equal(reference.begin(), reference.end(), outC.begin(),
                           [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}

TEST(beresnev_a_cannons_algorithm_mpi, Test_Random_2) {
  boost::mpi::communicator world;
  int n = 211;
  double dist = 100.0;
  std::vector<double> inA;
  std::vector<double> inB;
  std::vector<double> outC;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inA = getRandomVector(n * n, dist);
    inB = getRandomVector(n * n, dist);
    outC = std::vector<double>(n * n, 0.0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataPar->inputs_count.emplace_back(inA.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataPar->inputs_count.emplace_back(inB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&outC));
    taskDataPar->outputs_count.emplace_back(outC.size());
  }

  beresnev_a_cannons_algorithm_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference(n * n);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inA.data()));
    taskDataSeq->inputs_count.emplace_back(inA.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inB.data()));
    taskDataSeq->inputs_count.emplace_back(inB.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&reference));
    taskDataSeq->outputs_count.emplace_back(reference.size());

    beresnev_a_cannons_algorithm_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    for (size_t i = 0; i < reference.size(); ++i) {
      if (std::abs(reference[i] - outC[i]) >= 1e-9) {
        std::cerr << "Mismatch at index " << i << ": reference = " << reference[i] << ", outC = " << outC[i]
                  << ", diff = " << std::abs(reference[i] - outC[i]) << std::endl;
      }
    }
    ASSERT_TRUE(std::equal(reference.begin(), reference.end(), outC.begin(),
                           [](double a, double b) { return std::abs(a - b) < 1e-9; }));
  }
}