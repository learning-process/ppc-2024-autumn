// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/drozhdinov_d_gauss_vertical_scheme/include/ops_mpi.hpp"

namespace drozhdinov_d_gauss_vertical_scheme_mpi {
std::vector<double> genDenseMatrix(int n, int a) {
  std::vector<double> dense;
  std::vector<double> ed(n * n);
  std::vector<double> res(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n + i; j++) {
      dense.push_back(a + j);
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i < 2) {
        ed[j * n + i] = 0;
      } else if (i == j && i >= 2) {
        ed[j * n + i] = 1;
      } else {
        ed[j * n + i] = 0;
      }
    }
  }
  for (int i = 0; i < n * n; i++) {
    res[i] = (dense[i] + ed[i]);
  }
  return res;
}

std::vector<double> genElementaryMatrix(int rows, int columns) {
  std::vector<double> res;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (i == j) {
        res.push_back(1);
      } else {
        res.push_back(0);
      }
    }
  }
  return res;
}
template <typename T>
std::vector<T> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<T> vec(sz);
  vec[0] = gen() % 100;
  for (int i = 1; i < sz; i++) {
    vec[i] = (gen() % 100) - 49;
  }
  return vec;
}

template std::vector<int> drozhdinov_d_gauss_vertical_scheme_mpi::getRandomVector(int sz);
template std::vector<double> drozhdinov_d_gauss_vertical_scheme_mpi::getRandomVector(int sz);
}  // namespace drozhdinov_d_gauss_vertical_scheme_mpi

TEST(MPIGAUSS, WrongValidationTest) {
  boost::mpi::communicator world;
  int rows = 3;
  int columns = 3;
  std::vector<double> matrix = {1, 0, 0, 0, 1, 0, 0, 0, 0};  // det=0
  std::vector<double> b = {1, 1, 1};
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }
  if (world.rank() == 0) {
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
  // std::cout << world.rank();
}

TEST(MPIGAUSS, WrongValidationTest2) {
  boost::mpi::communicator world;
  int rows = 3;
  int columns = 3;
  std::vector<double> matrix = {1, 0, 0, 0, 1, 0, 0, 0, 0};
  std::vector<double> b = {1, 1, 1};
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    // taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }
  if (world.rank() == 0) {
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(MPIGAUSS, EquationTest) {
  boost::mpi::communicator world;
  int rows = 2;
  int columns = 2;
  std::vector<double> matrix = {1, 0, 0, 1};
  std::vector<double> b = {1, 1};
  std::vector<double> expres_par(rows);
  std::vector<double> res_par = {1, 1};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
    // std::cout << "aftret" << std::endl;
  }
}

TEST(MPIGAUSS, Equation2Test) {
  boost::mpi::communicator world;
  int rows = 3;
  int columns = 3;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_mpi::genDenseMatrix(rows, 1);
  std::vector<double> b = {1, 1, 1};
  std::vector<double> res_par = {-1, 1, 0};
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, Size100TestIdentity) {
  boost::mpi::communicator world;
  int rows = 10;
  int columns = 10;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_mpi::genElementaryMatrix(rows, columns);
  std::vector<double> b(rows, 1);
  std::vector<double> res_par(rows, 1);
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, Size100TestDense) {
  boost::mpi::communicator world;
  int rows = 10;
  int columns = 10;
  std::vector<int> a = drozhdinov_d_gauss_vertical_scheme_mpi::getRandomVector<int>(1);
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_mpi::genDenseMatrix(rows, *a.begin());
  std::vector<double> b(rows, 1);
  std::vector<double> res_par(rows, 0);
  res_par[0] = -1;
  res_par[1] = 1;
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, EmptyTest) {
  boost::mpi::communicator world;
  int rows = 0;
  int columns = 0;
  std::vector<double> matrix = {};
  std::vector<double> b = {};
  std::vector<double> expres_par(rows, 0);
  std::vector<double> res_par = {};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, Size10000TestIdentity) {
  boost::mpi::communicator world;
  int rows = 100;
  int columns = 100;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_mpi::genElementaryMatrix(rows, columns);
  std::vector<double> b(rows, 1);
  std::vector<double> res_par(rows, 1);
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
}

TEST(MPIGAUSS, RandomVectorB) {
  boost::mpi::communicator world;
  int rows = 10;
  int columns = 10;
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_mpi::genElementaryMatrix(rows, columns);
  std::vector<double> b = drozhdinov_d_gauss_vertical_scheme_mpi::getRandomVector<double>(rows);
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(b, expres_par);
  }
}

TEST(MPIGAUSS, Size10000TestDense) {
  boost::mpi::communicator world;
  int rows = 100;
  int columns = 100;
  std::vector<int> a = drozhdinov_d_gauss_vertical_scheme_mpi::getRandomVector<int>(1);
  std::vector<double> matrix = drozhdinov_d_gauss_vertical_scheme_mpi::genDenseMatrix(rows, *a.begin());
  std::vector<double> b(rows, 1);
  std::vector<double> res_par(rows, 0);
  res_par[0] = -1;
  res_par[1] = 1;
  std::vector<double> expres_par(rows);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs_count.emplace_back(b.size());
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_par.data()));
    taskDataPar->outputs_count.emplace_back(expres_par.size());
  }

  drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> expres_seq(rows);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
    taskDataSeq->inputs_count.emplace_back(matrix.size());
    taskDataSeq->inputs_count.emplace_back(b.size());
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres_seq.data()));
    taskDataSeq->outputs_count.emplace_back(expres_seq.size());

    // Create Task
    drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(expres_seq, expres_par);
    ASSERT_EQ(res_par, expres_par);
  }
  // std::cout << world.rank();
}
