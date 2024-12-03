#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sarafanov_m_gauss_jordan_method/include/ops_mpi.hpp"

namespace sarafanov_m_gauss_jordan_method_mpi {

std::vector<double> getRandomMatrix(int rows, int cols, double min = -20.0, double max = 20.0) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(min, max);
  std::vector<double> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

}  // namespace sarafanov_m_gauss_jordan_method_mpi

TEST(sarafanov_m_gauss_jordan_method_mpi, simple_three) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, 1, 10, 4, 8, 3, 20, 2, 5, 9, 30};
    n = 3;

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  bool parRunRes = taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<double> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential =
        std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodSequentialMPI>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    bool seqRunRes = taskSequential->run();
    taskSequential->post_processing();

    if (seqRunRes && parRunRes) {
      ASSERT_EQ(global_result.size(), seq_result.size());
      EXPECT_EQ(global_result, seq_result);
    } else {
      EXPECT_EQ(seqRunRes, parRunRes);
    }
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, simple_three_not_solve_at_1_iter) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {0,  2,  3,  4, 5,  6,  0,  8,  9,  10, 11, 12, 0,  14, 15,
                     16, 17, 18, 0, 20, 21, 22, 23, 24, 0,  26, 27, 28, 29, 30};
    n = 5;

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, simple_five_not_solve_at_2_iter) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    n = 5;

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, random_three) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 3;
    global_matrix = sarafanov_m_gauss_jordan_method_mpi::getRandomMatrix(n, n + 1);

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  bool isNonSingular = taskParallel->validation();
  if (isNonSingular) {
    taskParallel->pre_processing();
    bool parRunRes = taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      taskDataSeq->inputs_count.emplace_back(global_matrix.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      taskDataSeq->outputs_count.emplace_back(seq_result.size());

      auto taskSequential =
          std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      bool seqRunRes = taskSequential->run();
      taskSequential->post_processing();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, random_four) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 4;
    global_matrix = sarafanov_m_gauss_jordan_method_mpi::getRandomMatrix(n, n + 1, -30.0);

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  bool isNonSingular = taskParallel->validation();
  if (isNonSingular) {
    taskParallel->pre_processing();
    bool parRunRes = taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      taskDataSeq->inputs_count.emplace_back(global_matrix.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      taskDataSeq->outputs_count.emplace_back(seq_result.size());

      auto taskSequential =
          std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      bool seqRunRes = taskSequential->run();
      taskSequential->post_processing();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, random_five) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 5;
    global_matrix = sarafanov_m_gauss_jordan_method_mpi::getRandomMatrix(n, n + 1, -100.0, 200.0);

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  bool isNonSingular = taskParallel->validation();
  if (isNonSingular) {
    taskParallel->pre_processing();
    bool parRunRes = taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      taskDataSeq->inputs_count.emplace_back(global_matrix.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      taskDataSeq->outputs_count.emplace_back(seq_result.size());

      auto taskSequential =
          std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      bool seqRunRes = taskSequential->run();
      taskSequential->post_processing();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, random_six) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 6;
    global_matrix = sarafanov_m_gauss_jordan_method_mpi::getRandomMatrix(n, n + 1, -10.0, 10.0);

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  bool isNonSingular = taskParallel->validation();
  if (isNonSingular) {
    taskParallel->pre_processing();
    bool parRunRes = taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      taskDataSeq->inputs_count.emplace_back(global_matrix.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      taskDataSeq->outputs_count.emplace_back(seq_result.size());

      auto taskSequential =
          std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      bool seqRunRes = taskSequential->run();
      taskSequential->post_processing();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, random_seven) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 7;
    global_matrix = sarafanov_m_gauss_jordan_method_mpi::getRandomMatrix(n, n + 1, -7.0, 1000.0);

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  bool isNonSingular = taskParallel->validation();
  if (isNonSingular) {
    taskParallel->pre_processing();
    bool parRunRes = taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      taskDataSeq->inputs_count.emplace_back(global_matrix.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      taskDataSeq->outputs_count.emplace_back(seq_result.size());

      auto taskSequential =
          std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      bool seqRunRes = taskSequential->run();
      taskSequential->post_processing();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, random_ten) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 10;
    global_matrix = sarafanov_m_gauss_jordan_method_mpi::getRandomMatrix(n, n + 1);

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  bool isNonSingular = taskParallel->validation();
  if (isNonSingular) {
    taskParallel->pre_processing();
    bool parRunRes = taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      taskDataSeq->inputs_count.emplace_back(global_matrix.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      taskDataSeq->outputs_count.emplace_back(seq_result.size());

      auto taskSequential =
          std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      bool seqRunRes = taskSequential->run();
      taskSequential->post_processing();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(sarafanov_m_gauss_jordan_method_mpi, random_eleven) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 11;
    global_matrix = sarafanov_m_gauss_jordan_method_mpi::getRandomMatrix(n, n + 1);

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodParallelMPI>(taskDataPar);
  bool isNonSingular = taskParallel->validation();
  if (isNonSingular) {
    taskParallel->pre_processing();
    bool parRunRes = taskParallel->run();
    taskParallel->post_processing();

    if (world.rank() == 0) {
      std::vector<double> seq_result(global_result.size(), 0);

      auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
      taskDataSeq->inputs_count.emplace_back(global_matrix.size());

      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
      taskDataSeq->inputs_count.emplace_back(1);

      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
      taskDataSeq->outputs_count.emplace_back(seq_result.size());

      auto taskSequential =
          std::make_shared<sarafanov_m_gauss_jordan_method_mpi::GaussJordanMethodSequentialMPI>(taskDataSeq);
      ASSERT_TRUE(taskSequential->validation());
      taskSequential->pre_processing();
      bool seqRunRes = taskSequential->run();
      taskSequential->post_processing();

      if (seqRunRes && parRunRes) {
        ASSERT_EQ(global_result.size(), seq_result.size());
        EXPECT_EQ(global_result, seq_result);
      } else {
        EXPECT_EQ(seqRunRes, parRunRes);
      }
    }
  } else {
    EXPECT_TRUE(true);
  }
}
