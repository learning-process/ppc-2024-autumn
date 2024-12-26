#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/shurigin_lin_filtr_razbien_bloch_gaus_3x3/include/ops_mpi.hpp"

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_0x0) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 0;
  int num_cols = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(num_rows * num_cols);
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_1x1) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 1;
  int num_cols = 1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(num_rows * num_cols);
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_3x2) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 3;
  int num_cols = 2;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_matrix.resize(num_rows * num_cols);
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_3x3) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 3;
  int num_cols = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(0, 255);

    global_matrix.resize(num_rows * num_cols);
    for (int i = 0; i < num_rows * num_cols; i++) {
      global_matrix[i] = dist(gen);
    }
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskSeq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result.size(), seq_result.size());
    EXPECT_EQ(global_result, seq_result);
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_5x5) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 5;
  int num_cols = 5;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(0, 255);

    global_matrix.resize(num_rows * num_cols);
    for (int i = 0; i < num_rows * num_cols; i++) {
      global_matrix[i] = dist(gen);
    }
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskSeq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result.size(), seq_result.size());
    EXPECT_EQ(global_result, seq_result);
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_3x6) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 3;
  int num_cols = 6;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(0, 255);

    global_matrix.resize(num_rows * num_cols);
    for (int i = 0; i < num_rows * num_cols; i++) {
      global_matrix[i] = dist(gen);
    }
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskSeq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result.size(), seq_result.size());
    EXPECT_EQ(global_result, seq_result);
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_1x10) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 1;
  int num_cols = 10;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix.resize(num_rows * num_cols);
    global_result.resize(num_rows * num_cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }
  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_3x7) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 3;
  int num_cols = 7;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(0, 255);

    global_matrix.resize(num_rows * num_cols);
    for (int i = 0; i < num_rows * num_cols; i++) {
      global_matrix[i] = dist(gen);
    }
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskSeq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result.size(), seq_result.size());
    EXPECT_EQ(global_result, seq_result);
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_9x3) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 9;
  int num_cols = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(0, 255);

    global_matrix.resize(num_rows * num_cols);
    for (int i = 0; i < num_rows * num_cols; i++) {
      global_matrix[i] = dist(gen);
    }
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(global_matrix.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataSeq->inputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskSeq>(taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result.size(), seq_result.size());
    EXPECT_EQ(global_result, seq_result);
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_2x8) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 2;
  int num_cols = 8;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix.resize(num_rows * num_cols);
    global_result.resize(num_rows * num_cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }
  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, Test_11x1) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_result;
  int num_rows = 11;
  int num_cols = 1;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix.resize(num_rows * num_cols);
    global_result.resize(num_rows * num_cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }
  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, test_3x5_no_random_1) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::vector<int> global_matrix = {100, 150, 200, 250, 300, 120, 170, 220, 270, 320, 140, 190, 240, 290, 340};
  std::vector<int> global_result(15);
  int num_rows = 3;
  int num_cols = 5;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs = {reinterpret_cast<uint8_t*>(global_matrix.data()), reinterpret_cast<uint8_t*>(&num_rows),
                           reinterpret_cast<uint8_t*>(&num_cols)};
    taskDataPar->inputs_count = {static_cast<unsigned int>(global_matrix.size()), 1u, 1u};
    taskDataPar->outputs = {reinterpret_cast<uint8_t*>(global_result.data())};
    taskDataPar->outputs_count = {static_cast<unsigned int>(global_result.size())};
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result = {0, 0, 0, 0, 0, 0, 170, 220, 270, 0, 0, 0, 0, 0, 0};

    ASSERT_EQ(global_result.size(), expected_result.size());
    for (size_t i = 0; i < expected_result.size(); i++) {
      EXPECT_EQ(global_result[i], expected_result[i]) << "Mismatch at index " << i;
    }
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, test_3x5_no_random_2) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::vector<int> global_matrix = {100, 200, 300, 200, 100, 200, 300, 400, 300, 200, 100, 200, 300, 200, 100};
  std::vector<int> global_result(15);
  int num_rows = 3;
  int num_cols = 5;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs = {reinterpret_cast<uint8_t*>(global_matrix.data()), reinterpret_cast<uint8_t*>(&num_rows),
                           reinterpret_cast<uint8_t*>(&num_cols)};
    taskDataPar->inputs_count = {static_cast<unsigned int>(global_matrix.size()), 1u, 1u};
    taskDataPar->outputs = {reinterpret_cast<uint8_t*>(global_result.data())};
    taskDataPar->outputs_count = {static_cast<unsigned int>(global_result.size())};
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  if (world.rank() == 0) {
    std::vector<int> expected_result = {0, 0, 0, 0, 0, 0, 250, 300, 250, 0, 0, 0, 0, 0, 0};

    ASSERT_EQ(global_result.size(), expected_result.size());
    for (size_t i = 0; i < expected_result.size(); i++) {
      EXPECT_EQ(global_result[i], expected_result[i]) << "Mismatch at index " << i;
    }
  }
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, test_2x3_no_random) {
  boost::mpi::communicator world;
  std::vector<double> global_matrix = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> global_result(6, 0.0);
  int num_rows = 2;
  int num_cols = 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_rows));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_cols));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }
  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  if (world.rank() == 0) {
    EXPECT_FALSE(taskParallel->validation());
  } else {
    EXPECT_TRUE(taskParallel->validation());
  }
}
