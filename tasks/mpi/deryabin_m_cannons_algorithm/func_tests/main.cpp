#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

TEST(deryabin_m_cannons_algorithm_mpi, test_simple_matrix) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> input_matrix_B{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> output_matrix_C(9, 0);
  std::vector<double> true_solution{30, 36, 42, 66, 81, 96, 102, 126, 150};
  std::vector<double> output_x_vector_(10, 0);
  std::vector<std::vector<double>> out_matrix_C(1, output_matrix_C);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    taskDataPar->inputs_count.emplace_back(input_matrix_A.size());
    taskDataPar->inputs_count.emplace_back(input_matrix_B.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
    taskDataPar->outputs_count.emplace_back(out_matrix_C.size());
  }

  deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<std::vector<double>> reference_out_matrix_C(1, output_matrix_C);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_B.data()));
    taskDataSeq->inputs_count.emplace_back(input_matrix_A.size());
    taskDataSeq->inputs_count.emplace_back(input_matrix_B.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_matrix_C.data()));
    taskDataSeq->outputs_count.emplace_back(out_matrix_C.size());

    deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_out_matrix_C[0], out_matrix_C[0]);
  }
}
