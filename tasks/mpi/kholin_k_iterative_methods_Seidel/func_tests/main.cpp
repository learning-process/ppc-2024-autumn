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
  std::unique_ptr<float[]> in(new float[count_rows * count_colls]);
  std::unique_ptr<float[]> out(new float[count_rows]);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
    kholin_k_iterative_methods_Seidel_mpi::copyA_(in.get(), count_rows, count_colls);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.get()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }
  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  if (ProcRank == 0) {
    std::unique_ptr<float[]> out_ref(new float[count_rows]);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.get()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
  }
  kholin_k_iterative_methods_Seidel_mpi::freeA_();
}

 TEST(kholin_k_iterative_methods_Seidel_mpi, test_pre_processing) {
   int ProcRank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
   const size_t count_rows = 25;
   const size_t count_colls = 25;
   float epsilon = 0.001f;
   list_ops::ops_ op = list_ops::METHOD_SEIDEL;
   std::unique_ptr<float[]> in(new float[count_rows * count_colls]);
   std::unique_ptr<float[]> out(new float[count_rows]);
   // Create TaskData
   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

   if (ProcRank == 0) {
     kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
     kholin_k_iterative_methods_Seidel_mpi::copyA_(in.get(), count_rows, count_colls);
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
     taskDataPar->inputs_count.emplace_back(count_rows);
     taskDataPar->inputs_count.emplace_back(count_colls);
     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.get()));
     taskDataPar->outputs_count.emplace_back(count_rows);
   }

   kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
   testMpiTaskParallel.validation();
   ASSERT_EQ(testMpiTaskParallel.pre_processing(), true);

   if (ProcRank == 0) {
     std::unique_ptr<float[]> out_ref(new float[count_rows]);
     // Create TaskData
     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
     taskDataSeq->inputs_count.emplace_back(count_rows);
     taskDataSeq->inputs_count.emplace_back(count_colls);
     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.get()));
     taskDataSeq->outputs_count.emplace_back(count_rows);

     // Create Task
     kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
     testMpiTaskSequential.validation();
     ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
   }
   kholin_k_iterative_methods_Seidel_mpi::freeA_();
 }
//
// TEST(kholin_k_iterative_methods_Seidel_mpi, test_run) {
//   int ProcRank = 0;
//   MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
//   const size_t count_rows = 25;
//   const size_t count_colls = 25;
//   float epsilon = 0.001f;
//   list_ops::ops_ op = list_ops::METHOD_SEIDEL;
//   std::unique_ptr<float[]> in(new float[count_rows * count_colls]);
//   std::unique_ptr<float[]> out(new float[count_rows]);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//   if (ProcRank == 0) {
//     kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
//     kholin_k_iterative_methods_Seidel_mpi::copyA_(in.get(), count_rows, count_colls);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
//     taskDataPar->inputs_count.emplace_back(count_rows);
//     taskDataPar->inputs_count.emplace_back(count_colls);
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.get()));
//     taskDataPar->outputs_count.emplace_back(count_rows);
//   }
//
//   kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
//   testMpiTaskParallel.validation();
//   testMpiTaskParallel.pre_processing();
//   ASSERT_EQ(testMpiTaskParallel.run(), true);
//
//   if (ProcRank == 0) {
//     std::unique_ptr<float[]> out_ref(new float[count_rows]);
//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
//     taskDataSeq->inputs_count.emplace_back(count_rows);
//     taskDataSeq->inputs_count.emplace_back(count_colls);
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.get()));
//     taskDataSeq->outputs_count.emplace_back(count_rows);
//
//     // Create Task
//     kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
//     testMpiTaskSequential.validation();
//     testMpiTaskSequential.pre_processing();
//     ASSERT_EQ(testMpiTaskSequential.run(), true);
//   }
//   kholin_k_iterative_methods_Seidel_mpi::freeA_();
// }
//
// TEST(kholin_k_iterative_methods_Seidel_mpi, test_post_processing) {
//   int ProcRank = 0;
//   MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
//   const size_t count_rows = 25;
//   const size_t count_colls = 25;
//   float epsilon = 0.001f;
//   list_ops::ops_ op = list_ops::METHOD_SEIDEL;
//   std::unique_ptr<float[]> in(new float[count_rows * count_colls]);
//   std::unique_ptr<float[]> out(new float[count_rows]);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//   if (ProcRank == 0) {
//     kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
//     kholin_k_iterative_methods_Seidel_mpi::copyA_(in.get(), count_rows, count_colls);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
//     taskDataPar->inputs_count.emplace_back(count_rows);
//     taskDataPar->inputs_count.emplace_back(count_colls);
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.get()));
//     taskDataPar->outputs_count.emplace_back(count_rows);
//   }
//
//   kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
//   testMpiTaskParallel.validation();
//   testMpiTaskParallel.pre_processing();
//   testMpiTaskParallel.run();
//   ASSERT_EQ(testMpiTaskParallel.post_processing(), true);
//
//   if (ProcRank == 0) {
//     std::unique_ptr<float[]> out_ref(new float[count_rows]);
//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
//     taskDataSeq->inputs_count.emplace_back(count_rows);
//     taskDataSeq->inputs_count.emplace_back(count_colls);
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.get()));
//     taskDataSeq->outputs_count.emplace_back(count_rows);
//
//     // Create Task
//     kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
//     testMpiTaskSequential.validation();
//     testMpiTaskSequential.pre_processing();
//     testMpiTaskSequential.run();
//     ASSERT_EQ(testMpiTaskSequential.post_processing(), true);
//   }
//   kholin_k_iterative_methods_Seidel_mpi::freeA_();
// }
//
// TEST(kholin_k_iterative_methods_Seidel_mpi, validation_false_when_matrix_no_quadro) {
//   int ProcRank = 0;
//   MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
//   const size_t count_rows = 24;
//   const size_t count_colls = 25;
//   float epsilon = 0.001f;
//   list_ops::ops_ op = list_ops::METHOD_SEIDEL;
//   std::unique_ptr<float[]> in(new float[count_rows * count_colls]);
//   std::unique_ptr<float[]> out(new float[count_rows]);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//   if (ProcRank == 0) {
//     kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls);
//     kholin_k_iterative_methods_Seidel_mpi::copyA_(in.get(), count_rows, count_colls);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
//     taskDataPar->inputs_count.emplace_back(count_rows);
//     taskDataPar->inputs_count.emplace_back(count_colls);
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.get()));
//     taskDataPar->outputs_count.emplace_back(count_rows);
//   }
//
//   kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
//   ASSERT_EQ(testMpiTaskParallel.validation(), false);
//
//   if (ProcRank == 0) {
//     std::unique_ptr<float[]> out_ref(new float[count_rows]);
//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
//     taskDataSeq->inputs_count.emplace_back(count_rows);
//     taskDataSeq->inputs_count.emplace_back(count_colls);
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.get()));
//     taskDataSeq->outputs_count.emplace_back(count_rows);
//
//     // Create Task
//     kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
//     ASSERT_EQ(testMpiTaskSequential.validation(), false);
//   }
//   kholin_k_iterative_methods_Seidel_mpi::freeA_();
// }
////
// TEST(kholin_k_iterative_methods_Seidel_mpi, validation_false_when_matrix_no_diag) {
//   int ProcRank = 0;
//   MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
//   const size_t count_rows = 25;
//   const size_t count_colls = 25;
//   float epsilon = 0.001f;
//   list_ops::ops_ op = list_ops::METHOD_SEIDEL;
//   std::unique_ptr<float[]> in(new float[count_rows * count_colls]);
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//   bool IsValid = false;
//   if (ProcRank == 0) {
//     do {
//       int count = 0;
//       for (size_t i = 0; i < count_rows; i++) {
//         for (size_t j = 0; j < count_colls; j++) {
//           in[count_colls * i + j] = kholin_k_iterative_methods_Seidel_mpi::gen_float_value();
//         }
//         if (kholin_k_iterative_methods_Seidel_mpi::IsDiagPred(in.get(), count_colls, count_colls * i,
//                                                               count_colls * i + i)) {
//           count++;
//         }
//       }
//       if (count == count_rows) {
//         IsValid = true;
//       }
//     } while (IsValid);
//     kholin_k_iterative_methods_Seidel_mpi::setA_(in.get(), count_rows, count_colls);
//     std::unique_ptr<float[]> out(new float[count_rows]);
//     kholin_k_iterative_methods_Seidel_mpi::copyA_(in.get(), count_rows, count_colls);
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
//     taskDataPar->inputs_count.emplace_back(count_rows);
//     taskDataPar->inputs_count.emplace_back(count_colls);
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.get()));
//     taskDataPar->outputs_count.emplace_back(count_rows);
//   }
//
//   kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
//   ASSERT_EQ(testMpiTaskParallel.validation(), IsValid);
//
//   if (ProcRank == 0) {
//     std::unique_ptr<float[]> out_ref(new float[count_rows]);
//     // Create TaskData
//     std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.get()));
//     taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
//     taskDataSeq->inputs_count.emplace_back(count_rows);
//     taskDataSeq->inputs_count.emplace_back(count_colls);
//     taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.get()));
//     taskDataSeq->outputs_count.emplace_back(count_rows);
//
//     // Create Task
//     kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
//     ASSERT_EQ(testMpiTaskSequential.validation(), IsValid);
//   }
//   kholin_k_iterative_methods_Seidel_mpi::freeA_();
// }

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
