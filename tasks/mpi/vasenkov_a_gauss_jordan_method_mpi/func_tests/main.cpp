#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/vasenkov_a_gauss_jordan_method_mpi/include/GausJordanMpi.hpp"
#include "mpi/vasenkov_a_gauss_jordan_method_mpi/include/GausJordanSeq.hpp"

std::vector<double> RandomMatrix(int size) {
  std::vector<double> matrix(size * (size + 1));
  std::random_device rd;
  std::mt19937 gen(rd());
  double lowerLimit = -100.0;
  double upperLimit = 100.0;
  std::uniform_real_distribution<> dist(lowerLimit, upperLimit);

  for (int i = 0; i < size; ++i) {
    double row_sum = 0.0;
    double diag = (i * (size + 1) + i);
    for (int j = 0; j < size + 1; ++j) {
      if (i != j) {
        matrix[i * (size + 1) + j] = dist(gen);
        row_sum += std::abs(matrix[i * (size + 1) + j]);
      }
    }
    matrix[diag] = row_sum + 1;
  }

  return matrix;
}

TEST(vasenkov_a_gauss_jordan_method_mpi, simple_matrix_2x2) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 2;
    global_matrix = {1, 0, 5, 0, 1, 10};
    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
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

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
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

TEST(vasenkov_a_gauss_jordan_method_mpi, matrix_4x4) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 4;
    global_matrix = {1, 0, 0, 1, 10, 1, 2, 5, 10, 5, 1, 2, 5, 0, 5, 1, 2, 1, 10, 20};
    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
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

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
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

TEST(vasenkov_a_gauss_jordan_method_mpi, simple_random_matrix_3x3) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 3;
    global_matrix = RandomMatrix(n);
    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
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

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
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
TEST(vasenkov_a_gauss_jordan_method_mpi, simple_random_matrix_4x4) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 4;
    global_matrix = RandomMatrix(n);
    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
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

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
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
TEST(vasenkov_a_gauss_jordan_method_mpi, simple_random_matrix_2x2) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    n = 2;
    global_matrix = RandomMatrix(n);
    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
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

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
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

TEST(vasenkov_a_gauss_jordan_method_mpi, simple_matrix_3x3) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 0, 0, 2.8, 0, 1, 0, 2.6, 0, 0, 1, 0};
    n = 3;

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
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

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
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

TEST(vasenkov_a_gauss_jordan_method_mpi, matrix_3x3) {
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

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
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

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
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

TEST(vasenkov_a_gauss_jordan_method_mpi, matrix_3x3_under_zerro_data) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {2, -1, 1, 3, -3, -1, 2, -11, -2, 1, 2, -3};
    n = 3;

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
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

    auto taskSequential = std::make_shared<vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential>(taskDataSeq);
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
TEST(vasenkov_a_gauss_jordan_method_mpi, matrix_validation) {
  boost::mpi::communicator world;

  std::vector<double> global_matrix;
  int n;
  std::vector<double> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix = {1, 2, -3};
    n = 2;

    global_result.resize(n * (n + 1));

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<vasenkov_a_gauss_jordan_method_mpi::GaussJordanParallel>(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(taskParallel->validation());
  }
}