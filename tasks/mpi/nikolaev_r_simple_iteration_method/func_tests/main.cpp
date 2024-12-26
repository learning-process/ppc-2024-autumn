#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/nikolaev_r_simple_iteration_method/include/ops_mpi.hpp"

TEST(nikolaev_r_simple_iteration_method_mpi, test_2x2_matrix) {
  boost::mpi::communicator world;
  const size_t m_size = 2;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {5.0, 2.0, 1.0, 6.0};
  std::vector<double> vect = {-1.0, 5.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_seq.data()));
    taskDataSeq->inputs_count.emplace_back(in_seq.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataSeq->inputs_count.emplace_back(matr.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataSeq->inputs_count.emplace_back(vect.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodSequential simpleIterationSeq(taskDataSeq);
    ASSERT_TRUE(simpleIterationSeq.validation());
    ASSERT_TRUE(simpleIterationSeq.pre_processing());
    ASSERT_TRUE(simpleIterationSeq.run());
    ASSERT_TRUE(simpleIterationSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }
  nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
  ASSERT_TRUE(simpleIterationPar.validation());
  ASSERT_TRUE(simpleIterationPar.pre_processing());
  ASSERT_TRUE(simpleIterationPar.run());
  ASSERT_TRUE(simpleIterationPar.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out_seq[i], out_par[i], 1e-6);
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_3x3_matrix) {
  boost::mpi::communicator world;
  const size_t m_size = 3;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {10.0, 2.0, 3.0, 5.0, 15.0, 7.0, 8.0, 1.0, 12.0};
  std::vector<double> vect = {2.0, -18.0, 1.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_seq.data()));
    taskDataSeq->inputs_count.emplace_back(in_seq.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataSeq->inputs_count.emplace_back(matr.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataSeq->inputs_count.emplace_back(vect.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodSequential simpleIterationSeq(taskDataSeq);
    ASSERT_TRUE(simpleIterationSeq.validation());
    ASSERT_TRUE(simpleIterationSeq.pre_processing());
    ASSERT_TRUE(simpleIterationSeq.run());
    ASSERT_TRUE(simpleIterationSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }
  nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
  ASSERT_TRUE(simpleIterationPar.validation());
  ASSERT_TRUE(simpleIterationPar.pre_processing());
  ASSERT_TRUE(simpleIterationPar.run());
  ASSERT_TRUE(simpleIterationPar.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out_seq[i], out_par[i], 1e-6);
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_4x4_matrix) {
  boost::mpi::communicator world;
  const size_t m_size = 4;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {10.0, 2.0, 3.0, -4.0, 5.0, 17.0, 5.0, 2.0, 12.0, 4.0, 20.0, 2.0, 6.0, -2.0, 1.0, 14.0};
  std::vector<double> vect = {5.0, 8.0, -1.0, 3.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_seq.data()));
    taskDataSeq->inputs_count.emplace_back(in_seq.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataSeq->inputs_count.emplace_back(matr.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataSeq->inputs_count.emplace_back(vect.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodSequential simpleIterationSeq(taskDataSeq);
    ASSERT_TRUE(simpleIterationSeq.validation());
    ASSERT_TRUE(simpleIterationSeq.pre_processing());
    ASSERT_TRUE(simpleIterationSeq.run());
    ASSERT_TRUE(simpleIterationSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }
  nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
  ASSERT_TRUE(simpleIterationPar.validation());
  ASSERT_TRUE(simpleIterationPar.pre_processing());
  ASSERT_TRUE(simpleIterationPar.run());
  ASSERT_TRUE(simpleIterationPar.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out_seq[i], out_par[i], 1e-6);
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_5x5_matrix) {
  boost::mpi::communicator world;
  const size_t m_size = 5;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {23.0,  2.0, 3.0,  -5.0, 5.0,  -2.0, 25.0, 7.0, -2.0, 1.0,  1.0,  -2.0, 36.0,
                              -12.0, 5.0, 14.0, 8.0,  12.0, 45.0, -9.0, 3.0, 5.0,  -4.0, 12.0, 56.0};
  std::vector<double> vect = {5.0, 18.0, -1.0, -34.0, 2.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_seq.data()));
    taskDataSeq->inputs_count.emplace_back(in_seq.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataSeq->inputs_count.emplace_back(matr.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataSeq->inputs_count.emplace_back(vect.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodSequential simpleIterationSeq(taskDataSeq);
    ASSERT_TRUE(simpleIterationSeq.validation());
    ASSERT_TRUE(simpleIterationSeq.pre_processing());
    ASSERT_TRUE(simpleIterationSeq.run());
    ASSERT_TRUE(simpleIterationSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }
  nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
  ASSERT_TRUE(simpleIterationPar.validation());
  ASSERT_TRUE(simpleIterationPar.pre_processing());
  ASSERT_TRUE(simpleIterationPar.run());
  ASSERT_TRUE(simpleIterationPar.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out_seq[i], out_par[i], 1e-6);
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_identity_matrix) {
  boost::mpi::communicator world;
  const size_t m_size = 3;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> vect = {1.0, -1.0, 1.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_seq.data()));
    taskDataSeq->inputs_count.emplace_back(in_seq.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataSeq->inputs_count.emplace_back(matr.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataSeq->inputs_count.emplace_back(vect.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_seq.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodSequential simpleIterationSeq(taskDataSeq);
    ASSERT_TRUE(simpleIterationSeq.validation());
    ASSERT_TRUE(simpleIterationSeq.pre_processing());
    ASSERT_TRUE(simpleIterationSeq.run());
    ASSERT_TRUE(simpleIterationSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }
  nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
  ASSERT_TRUE(simpleIterationPar.validation());
  ASSERT_TRUE(simpleIterationPar.pre_processing());
  ASSERT_TRUE(simpleIterationPar.run());
  ASSERT_TRUE(simpleIterationPar.post_processing());

  for (size_t i = 0; i < m_size; i++) {
    ASSERT_NEAR(out_seq[i], out_par[i], 1e-6);
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_non_diagonally_dominant_matrix) {
  boost::mpi::communicator world;
  const size_t m_size = 3;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {4.0, 5.0, 1.0, 3.0, 7.0, -2.0, -2.0, 3.0, 8.0};
  std::vector<double> vect = {2.0, -18.0, 1.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
    ASSERT_FALSE(simpleIterationPar.validation());
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_singular_matrix) {
  boost::mpi::communicator world;
  const size_t m_size = 3;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {2.0, 4.0, 6.0, 1.0, 2.0, 3.0, 3.0, 6.0, 9.0};
  std::vector<double> vect = {2.0, -18.0, 1.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
    ASSERT_FALSE(simpleIterationPar.validation());
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_incorrect_matrix_size) {
  boost::mpi::communicator world;
  const size_t m_size = 0;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {2.0, 4.0, 6.0, 1.0, 2.0, 3.0, 3.0, 6.0, 9.0};
  std::vector<double> vect = {2.0, -18.0, 1.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
    ASSERT_FALSE(simpleIterationPar.validation());
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_incorrect_input_data_matr) {
  boost::mpi::communicator world;
  const size_t m_size = 3;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {2.0, 4.0, 6.0, 1.0, 2.0, 3.0, 3.0, 6.0, 9.0};
  std::vector<double> vect = {2.0, -18.0, 1.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
    ASSERT_FALSE(simpleIterationPar.validation());
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_incorrect_input_data_vect) {
  boost::mpi::communicator world;
  const size_t m_size = 3;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {2.0, 4.0, 6.0, 1.0, 2.0, 3.0, 3.0, 6.0, 9.0};
  std::vector<double> vect = {2.0, -18.0, 1.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
    ASSERT_FALSE(simpleIterationPar.validation());
  }
}

TEST(nikolaev_r_simple_iteration_method_mpi, test_incorrect_output_data) {
  boost::mpi::communicator world;
  const size_t m_size = 3;
  std::vector<size_t> in_seq(1, m_size);
  std::vector<size_t> in_par(1, m_size);
  std::vector<double> matr = {2.0, 4.0, 6.0, 1.0, 2.0, 3.0, 3.0, 6.0, 9.0};
  std::vector<double> vect = {2.0, -18.0, 1.0};
  std::vector<double> out_seq(m_size, 0.0);
  std::vector<double> out_par(m_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_par.data()));
    taskDataPar->inputs_count.emplace_back(in_par.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matr.data()));
    taskDataPar->inputs_count.emplace_back(matr.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(vect.data()));
    taskDataPar->inputs_count.emplace_back(vect.size());

    nikolaev_r_simple_iteration_method_mpi::SimpleIterationMethodParallel simpleIterationPar(taskDataPar);
    ASSERT_FALSE(simpleIterationPar.validation());
  }
}