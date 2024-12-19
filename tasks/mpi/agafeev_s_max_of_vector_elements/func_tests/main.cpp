#include <gtest/gtest.h>

#include <iostream>

#include "boost/mpi/communicator.hpp"
#include "mpi/agafeev_s_max_of_vector_elements/include/ops_mpi.hpp"

template <typename T>
static std::vector<T> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(std::time(nullptr));
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); ++i) matrix[i] = rand_gen() % 200 - 100;

  return matrix;
}

TEST(agafeev_s_max_of_vector_elements, test_empty_matrix) {
  boost::mpi::communicator world;
  std::vector<int> in_matrix(0);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<int>(0, 0);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);

  if (world.rank() == 0) {
    ASSERT_EQ(testTask->validation(), false);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_1x1_matrix) {
  boost::mpi::communicator world;
  std::vector<int> in_matrix(9);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<int>(1, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(in_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    auto testTaskSeq = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq<int>>(taskDataSeq);
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

    int right_answer = std::numeric_limits<int>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
    ASSERT_EQ(right_answer, seq_out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_3x3_matrix) {
  boost::mpi::communicator world;
  std::vector<int> in_matrix(9);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<int>(3, 3);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(in_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    auto testTaskSeq = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq<int>>(taskDataSeq);
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

    int right_answer = std::numeric_limits<int>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
    ASSERT_EQ(right_answer, seq_out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_100x100_matrix) {
  boost::mpi::communicator world;

  std::vector<int> in_matrix(10000);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<int>(100, 100);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(in_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    auto testTaskSeq = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq<int>>(taskDataSeq);
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

    int right_answer = std::numeric_limits<int>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
    ASSERT_EQ(right_answer, seq_out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_1000x12_matrix) {
  boost::mpi::communicator world;

  std::vector<int> in_matrix(12000);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<int>(1000, 12);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskData->inputs_count.emplace_back(in_matrix.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<int>>(taskData);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    std::vector<int> seq_out(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(in_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    auto testTaskSeq = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq<int>>(taskDataSeq);
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

    int right_answer = std::numeric_limits<int>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
    ASSERT_EQ(right_answer, seq_out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_100x100_matrix_double) {
  boost::mpi::communicator world;

  std::vector<double> in_matrix(100 * 100);
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<double>(100, 100);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataMpi->inputs_count.emplace_back(in_matrix.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMpi->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<double>>(taskDataMpi);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    std::vector<double> seq_out(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(in_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    auto testTaskSeq = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq<double>>(taskDataSeq);
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

    double right_answer = std::numeric_limits<double>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
    ASSERT_EQ(right_answer, seq_out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_9x45_matrix_double) {
  boost::mpi::communicator world;

  std::vector<double> in_matrix(9 * 45);
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<double>(9, 45);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataMpi->inputs_count.emplace_back(in_matrix.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMpi->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<double>>(taskDataMpi);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    std::vector<double> seq_out(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(in_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    auto testTaskSeq = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq<double>>(taskDataSeq);
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

    double right_answer = std::numeric_limits<double>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
    ASSERT_EQ(right_answer, seq_out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, test_find_in_300x200_matrix_double) {
  boost::mpi::communicator world;

  std::vector<double> in_matrix(9 * 45);
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = create_RandomMatrix<double>(9, 45);
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataMpi->inputs_count.emplace_back(in_matrix.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMpi->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<double>>(taskDataMpi);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    std::vector<double> seq_out(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(in_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    auto testTaskSeq = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq<double>>(taskDataSeq);
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

    double right_answer = std::numeric_limits<double>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
    ASSERT_EQ(right_answer, seq_out[0]);
  }
}

TEST(agafeev_s_max_of_vector_elements, negative_numbers_test) {
  boost::mpi::communicator world;

  std::vector<double> in_matrix(3 * 10);
  std::vector<double> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in_matrix = {-20, -93, -93, -31, -56, -58, -16, -41, -88, -87, -35, -24, -4, -83, -54,
                 -93, -16, -44, -95, -87, -37, -15, -42, -82, -88, -18, -22, -2, -88, -94};
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataMpi->inputs_count.emplace_back(in_matrix.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMpi->outputs_count.emplace_back(out.size());
  }

  auto testTask = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixMpi<double>>(taskDataMpi);
  bool isValid = testTask->validation();
  ASSERT_EQ(isValid, true);
  testTask->pre_processing();
  testTask->run();
  testTask->post_processing();

  if (world.rank() == 0) {
    std::vector<double> seq_out(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_matrix.data()));
    taskDataSeq->inputs_count.emplace_back(in_matrix.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_out.data()));
    taskDataSeq->outputs_count.emplace_back(seq_out.size());

    auto testTaskSeq = std::make_shared<agafeev_s_max_of_vector_elements_mpi::MaxMatrixSeq<double>>(taskDataSeq);
    ASSERT_EQ(testTaskSeq->validation(), true);
    testTaskSeq->pre_processing();
    testTaskSeq->run();
    testTaskSeq->post_processing();

    double right_answer = std::numeric_limits<double>::min();
    for (auto &&t : in_matrix)
      if (right_answer < t) right_answer = t;

    ASSERT_EQ(right_answer, out[0]);
    ASSERT_EQ(right_answer, seq_out[0]);
  }
}