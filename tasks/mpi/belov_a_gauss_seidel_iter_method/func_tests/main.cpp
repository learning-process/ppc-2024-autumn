#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/belov_a_gauss_seidel_iter_method/include/ops_mpi.hpp"

namespace belov_a_gauss_seidel_mpi {
std::vector<double> generateDiagonallyDominantMatrix(int n) {
  std::vector<double> A_local(n * n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A_local[i * n + j] = dis(gen);
        row_sum += abs(A_local[i * n + j]);
      }
    }
    A_local[i * n + i] = row_sum + abs(dis(gen)) + 1.0;
  }
  return A_local;
}

std::vector<double> generateFreeMembers(int n) {
  std::vector<double> freeMembers(n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    freeMembers[i] = dis(gen);
  }
  return freeMembers;
}
}  // namespace belov_a_gauss_seidel_mpi

using namespace belov_a_gauss_seidel_mpi;

TEST(belov_a_gauss_seidel_mpi, Test_3x3_Predefined_Matrix) {
  boost::mpi::communicator world;

  int n = 3;
  double epsilon = 0.1;
  std::vector<double> matrix = {10, -3, 2, 3, -10, -2, 2, -3, 10};
  std::vector<double> freeMembers = {10, -23, 26};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_4x4_Generated_Matrix) {
  boost::mpi::communicator world;

  int n = 4;
  double epsilon = 0.2;
  std::vector<double> matrix = generateDiagonallyDominantMatrix(n);
  std::vector<double> freeMembers = generateFreeMembers(n);
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_4x4_Diagonally_Dominant_Predef_Matrix) {
  boost::mpi::communicator world;

  int n = 4;
  double epsilon = 0.1;
  std::vector<double> matrix = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8};
  std::vector<double> freeMembers = {6, 25, -11, 15};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_Negative_Values_Only) {
  boost::mpi::communicator world;

  int n = 3;
  double epsilon = 0.1;
  std::vector<double> matrix = {-9, -5, -1, -4, -11, -3, -2, -5, -16};
  std::vector<double> freeMembers = {-1, -12, -9};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_2x2_Double_Matrix) {
  boost::mpi::communicator world;

  int n = 2;
  double epsilon = 0.1;
  std::vector<double> matrix = {5.0, -2.0, -1.0, 4.0};
  std::vector<double> freeMembers = {3.0, 7.0};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_3x3_Matrix) {
  boost::mpi::communicator world;

  int n = 3;
  double epsilon = 0.1;
  std::vector<double> matrix = {10.0, -1.0, 2.0, -1.0, 11.0, -1.0, 2.0, -1.0, 10.0};
  std::vector<double> freeMembers = {6.0, 25.0, -11.0};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_Large_10x10_Generated_Matrix) {
  boost::mpi::communicator world;

  int n = 10;
  double epsilon = 0.2;
  std::vector<double> matrix = generateDiagonallyDominantMatrix(n);
  std::vector<double> freeMembers = generateFreeMembers(n);
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_Large_Epsilon) {
  boost::mpi::communicator world;

  int n = 3;
  double epsilon = 1.0;
  std::vector<double> matrix = {5, 1, 0, 1, 5, 1, 0, 1, 5};
  std::vector<double> freeMembers = {6, 7, 8};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_Small_Epsilon) {
  boost::mpi::communicator world;

  int n = 3;
  double epsilon = 1e-3;
  std::vector<double> matrix = {6, -1, -1, -1, 6, -1, -1, -1, 6};
  std::vector<double> freeMembers = {11.33, 32.00, 42.00};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());
  }

  GaussSeidelParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    std::vector<double> solutionSeq(n, 0);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs_count.emplace_back(freeMembers.size());
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    GaussSeidelSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    for (int i = 0; i < n; ++i) {
      ASSERT_NEAR(solutionMpi[i], solutionSeq[i], epsilon);
    }
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_Invalid_Matrix_Size) {
  boost::mpi::communicator world;

  int n = 3;
  double epsilon = 0.1;
  std::vector<double> matrix = {2, 5, 1, -7, 12, 8};
  std::vector<double> freeMembers = {5, -2};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());

    GaussSeidelParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_Non_Square_Matrix) {
  boost::mpi::communicator world;

  int n = 3;
  double epsilon = 0.1;
  std::vector<double> matrix = {1, 2, 3, 4, 5, 6};  // non square 2x3 matrix
  std::vector<double> freeMembers = {7, 8};
  std::vector<double> solutionMpi(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);  // emplace back incorrect rows numbers
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());

    GaussSeidelParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_No_Diagonal_Dominance) {
  boost::mpi::communicator world;

  int n = 3;
  double epsilon = 0.1;
  std::vector<double> matrix = {2, 3, 1, 1, 2, 3, 3, 1, 2};
  std::vector<double> freeMembers = {5, 6, 7};
  std::vector<double> solutionMpi(n, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(freeMembers.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs_count.emplace_back(freeMembers.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionMpi.data()));
    taskDataPar->outputs_count.emplace_back(solutionMpi.size());

    GaussSeidelParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_mpi_small_difference) {
  double a = 1.0;
  double b = 1.0 + std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(GaussSeidelParallel::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_mpi_large_difference) {
  double a = 1.0;
  double b = 2.0;
  EXPECT_FALSE(GaussSeidelParallel::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_mpi_negative_numbers) {
  double a = -1.0;
  double b = -1.0000001;
  EXPECT_TRUE(GaussSeidelParallel::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_mpi_zero_numbers) {
  double a = 0.0;
  double b = 0.0;
  EXPECT_TRUE(GaussSeidelParallel::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_mpi_mixed_sign_numbers) {
  double a = -1.0;
  double b = 1.0;
  EXPECT_FALSE(GaussSeidelParallel::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_seq_equal_numbers) {
  double a = 1.0;
  double b = 1.0;
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_seq_small_difference) {
  double a = 1.0;
  double b = 1.0 + std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_seq_large_difference) {
  double a = 10.0;
  double b = 5.0;
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_seq_negative_numbers) {
  double a = -1.0;
  double b = -1.0 - std::numeric_limits<double>::epsilon();
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}

TEST(belov_a_gauss_seidel_mpi, Test_ge_double_seq_zero_numbers) {
  double a = 0.0;
  double b = 0.0;
  EXPECT_TRUE(GaussSeidelSequential::ge_double(a, b));
}