#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/deryabin_m_jacobi_iterative_method/include/ops_mpi.hpp"

TEST(deryabin_m_jacobi_iterative_method_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_{1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                    0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};
  std::vector<double> input_right_vector_{1, 2, 3, 4, 5, 6};
  std::vector<double> output_x_vector_ = std::vector<double>(6, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar =
      std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix_.size());
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel
      testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential
        testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi, test_triangular_matrix) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_{
      16, 1, 2, 3,  4,  5,  0, 31, 6, 7, 8,  9,  0, 0, 34, 10, 11, 12,
      0,  0, 0, 28, 13, 14, 0, 0,  0, 0, 16, 15, 0, 0, 0,  0,  0,  17};
  std::vector<double> input_right_vector_{86, 202, 269, 261, 170, 102};
  std::vector<double> output_x_vector_ = std::vector<double>(6, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar =
      std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix_.size());
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel
      testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential
        testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi,
     test_diagonal_elements_are_much_larger_than_non_diagonal) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_{999, 1,   2,  3,   4,  5,   6,  999, 7,
                                    8,   9,   10, 11,  12, 999, 13, 14,  15,
                                    16,  17,  18, 999, 19, 20,  21, 22,  23,
                                    24,  999, 25, 26,  27, 28,  29, 30,  999};
  std::vector<double> input_right_vector_{1069, 2162, 3244, 4315, 5375, 6424};
  std::vector<double> output_x_vector_ = std::vector<double>(6, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar =
      std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix_.size());
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel
      testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential
        testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi, invalid_matrix_zeros_on_diagonal) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_{
      0,  1,  2,  3, 4,  5,  6,  0,  7,  8,  9, 10, 11, 12, 0,  13, 14, 15,
      16, 17, 18, 0, 19, 20, 21, 22, 23, 24, 0, 25, 26, 27, 28, 29, 30, 0};
  std::vector<double> input_right_vector_{70, 164, 247, 319, 380, 430};
  std::vector<double> output_x_vector_ = std::vector<double>(6, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar =
      std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix_.size());
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel
      testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential
        testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi,
     invalid_matrix_non_strict_diaganol_predominance) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_{
      15, 1,  2,  3,  4,  5,  6,  40, 7,  8,  9,   10, 11, 12, 65, 13, 14, 15,
      16, 17, 18, 90, 19, 20, 21, 22, 23, 24, 115, 25, 26, 27, 28, 29, 30, 140};
  std::vector<double> input_right_vector_{85, 244, 442, 679, 955, 1270};
  std::vector<double> output_x_vector_ = std::vector<double>(6, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);

  std::shared_ptr<ppc::core::TaskData> taskDataPar =
      std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix_.size());
    taskDataPar->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel
      testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_x_vec(1, output_x_vector_);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq =
        std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_.size());
    taskDataSeq->inputs.emplace_back(
        reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataSeq->inputs_count.emplace_back(input_right_vector_.size());
    taskDataSeq->outputs.emplace_back(
        reinterpret_cast<uint8_t*>(reference_out_x_vec.data()));
    taskDataSeq->outputs_count.emplace_back(reference_out_x_vec.size());

    deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskSequential
        testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_x_vec[0], out_x_vec[0]);
  }
}
