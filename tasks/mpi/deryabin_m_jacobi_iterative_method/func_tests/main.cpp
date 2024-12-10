#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/deryabin_m_jacobi_iterative_method/include/ops_mpi.hpp"

TEST(deryabin_m_jacobi_iterative_method_mpi, test_random_valid_matrix) {
  boost::mpi::communicator world;
  // std::random_device rd;
  std::default_random_engine gen(rand());
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(1, 10);
  std::uniform_real_distribution<> distribut(91, 100);
  std::vector<double> input_matrix_(100, distrib(gen));
  std::vector<double> input_right_vector_(10, distrib(gen));
  for (unsigned short i = 0; i < 10; i++) {
    input_matrix_[i * 11] = distribut(gen);
  }
  std::vector<double> output_x_vector_ = std::vector<double>(10, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi, test_random_3X_diagonal_matrix) {
  boost::mpi::communicator world;
  // std::random_device rd;
  std::default_random_engine gen(rand());
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(1, 10);
  std::uniform_real_distribution<> distribut(91, 100);
  std::vector<double> input_matrix_(100, 0);
  std::vector<double> input_right_vector_(10, distrib(gen));
  for (unsigned short i = 0; i < 10; i++) {
    input_matrix_[i * 11] = distribut(gen);
    if (i != 0) {
      input_matrix_[i * 11 - 1] = distrib(gen);
    }
    if (i != 9) {
      input_matrix_[i * 11 + 1] = distrib(gen);
    }
  }
  std::vector<double> output_x_vector_ = std::vector<double>(10, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi, test_random_1X_diagonal_matrix) {
  boost::mpi::communicator world;
  // std::random_device rd;
  std::default_random_engine gen(rand());
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<> distribut(1, 100);
  std::vector<double> input_matrix_(100, 0);
  std::vector<double> input_right_vector_(10, distribut(gen));
  for (unsigned short i = 0; i < 10; i++) {
    input_matrix_[i * 11] = distribut(gen);
  }
  std::vector<double> output_x_vector_ = std::vector<double>(10, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi, test_random_diagonal_elements_are_much_larger_than_non_diagonal) {
  boost::mpi::communicator world;
  // std::random_device rd;
  std::default_random_engine gen(rand());
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(1, 10);
  std::vector<double> input_matrix_(100, distrib(gen));
  std::vector<double> input_right_vector_(10, distrib(gen));
  for (unsigned short i = 0; i < 10; i++) {
    input_matrix_[i * 11] = 100 * distrib(gen);
  }
  std::vector<double> output_x_vector_ = std::vector<double>(10, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi, random_invalid_matrix_zeros_on_diagonal) {
  boost::mpi::communicator world;
  // std::random_device rd;
  std::default_random_engine gen(rand());
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(1, 10);
  std::vector<double> input_matrix_(100, distrib(gen));
  std::vector<double> input_right_vector_(10, distrib(gen));
  for (unsigned short i = 0; i < 10; i++) {
    input_matrix_[i * 11] = 0;
  }
  std::vector<double> output_x_vector_ = std::vector<double>(10, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}
