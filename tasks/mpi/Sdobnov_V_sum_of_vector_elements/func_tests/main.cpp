#include <gtest/gtest.h>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include "mpi/Sdobnov_V_sum_of_vector_elements/include/ops_mpi.hpp"

TEST(Sdobnov_V_sum_of_vector_elements_par, EmptyInput) {
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_sum_of_vector_elements_par, EmptyOutput) {
  boost::mpi::communicator world;
  int rows = 10;
  int columns = 10;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
    }
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  ASSERT_FALSE(test.validation());
}

TEST(Sdobnov_V_sum_of_vector_elements_par, EmptyMatrix) {
  boost::mpi::communicator world;
  int rows = 0;
  int columns = 0;
  int res;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);

  ASSERT_TRUE(test.validation());
  test.pre_processing();
  test.run();
  test.post_processing();
  ASSERT_EQ(res, 0);
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix10x10) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 10;
  int res = 0;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(res, respar);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix100x100) {
  boost::mpi::communicator world;

  int rows = 100;
  int columns = 100;
  int res = 0;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(res, respar);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix100x10) {
  boost::mpi::communicator world;

  int rows = 100;
  int columns = 10;
  int res = 0;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(res, respar);
  }
}

TEST(Sdobnov_V_sum_of_vector_elements_par, Matrix10x100) {
  boost::mpi::communicator world;

  int rows = 10;
  int columns = 100;
  int res = 0;
  std::vector<std::vector<int>> input = Sdobnov_V_sum_of_vector_elements::generate_random_matrix(rows, columns);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    for (int i = 0; i < input.size(); i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  }

  Sdobnov_V_sum_of_vector_elements::SumVecElemParallel test(taskDataPar);
  ASSERT_EQ(test.validation(), true);
  test.pre_processing();
  test.run();
  test.post_processing();

  if (world.rank() == 0) {
    int respar = res;
    Sdobnov_V_sum_of_vector_elements::SumVecElemSequential testseq(taskDataPar);
    testseq.validation();
    testseq.pre_processing();
    testseq.run();
    testseq.post_processing();
    ASSERT_EQ(res, respar);
  }
}
