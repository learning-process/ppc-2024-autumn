#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kholin_k_iterative_methods_Seidel/include/ops_mpi.hpp"
//
TEST(kholin_k_iterative_methods_Seidel_mpi, validation_true_when_matrix_with_diag_pred) {
  int ProcRank = 0;
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  float *in = nullptr;
  float *out = nullptr;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    in = new float[count_rows * count_colls];
    out = new float[count_rows];
    kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
    kholin_k_iterative_methods_Seidel_mpi::copyA_(in, count_rows, count_colls);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }
  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  float *out_ref = nullptr;
  if (ProcRank == 0) {
    out_ref = new float[count_rows];
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
  }
  if (ProcRank == 0) {
    delete[] in;
    delete[] out;
    delete[] out_ref;
  }
  if (ProcRank == 0) {
    kholin_k_iterative_methods_Seidel_mpi::freeA_();
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, test_pre_processing) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  float *in = nullptr;
  float *out = nullptr;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    in = new float[count_rows * count_colls];
    out = new float[count_rows];
    kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
    kholin_k_iterative_methods_Seidel_mpi::copyA_(in, count_rows, count_colls);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  ASSERT_EQ(testMpiTaskParallel.pre_processing(), true);

  float *out_ref = nullptr;
  if (ProcRank == 0) {
    out_ref = new float[count_rows];
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
  }
  if (ProcRank == 0) {
    delete[] in;
    delete[] out;
    delete[] out_ref;
  }
  if (ProcRank == 0) {
    kholin_k_iterative_methods_Seidel_mpi::freeA_();
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, test_run) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  float *in = nullptr;
  float *out = nullptr;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    in = new float[count_rows * count_colls];
    out = new float[count_rows];
    kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
    kholin_k_iterative_methods_Seidel_mpi::copyA_(in, count_rows, count_colls);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  ASSERT_EQ(testMpiTaskParallel.run(), true);

  float *out_ref = nullptr;
  if (ProcRank == 0) {
    out_ref = new float[count_rows];
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    ASSERT_EQ(testMpiTaskSequential.run(), true);
  }
  if (ProcRank == 0) {
    delete[] in;
    delete[] out;
    delete[] out_ref;
  }
  if (ProcRank == 0) {
    kholin_k_iterative_methods_Seidel_mpi::freeA_();
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, test_post_processing) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  float *in = nullptr;
  float *out = nullptr;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    in = new float[count_colls * count_rows];
    out = new float[count_rows];
    kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
    kholin_k_iterative_methods_Seidel_mpi::copyA_(in, count_rows, count_colls);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  ASSERT_EQ(testMpiTaskParallel.post_processing(), true);

  float *out_ref = nullptr;
  if (ProcRank == 0) {
    out_ref = new float[count_rows];
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);
  }
  if (ProcRank == 0) {
    delete[] in;
    delete[] out;
    delete[] out_ref;
  }
  if (ProcRank == 0) {
    kholin_k_iterative_methods_Seidel_mpi::freeA_();
  }
}

// int main(int argc, char **argv) {
//   boost::mpi::environment env(argc, argv);
//   boost::mpi::communicator world;
//   ::testing::InitGoogleTest(&argc, argv);
//   ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
//   if (world.rank() != 0) {
//     delete listeners.Release(listeners.default_result_printer());
//   }
//   return RUN_ALL_TESTS();
// }
