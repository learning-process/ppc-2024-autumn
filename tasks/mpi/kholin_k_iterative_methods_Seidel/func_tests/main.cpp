#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <mpi/kholin_k_iterative_methods_Seidel/src/ops_mpi.cpp>

#include "mpi/kholin_k_iterative_methods_Seidel/include/ops_mpi.hpp"
//
TEST(kholin_k_iterative_methods_Seidel_mpi, validation_true_when_matrix_with_diag_pred) {
  int ProcRank = 0;
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    float p1 = -(1000.0f * 1000.0f * 1000.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }
  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  std::vector<float> out_ref;
  if (ProcRank == 0) {
    out_ref = std::vector<float>(count_rows);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, test_pre_processing) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    float p1 = -(1000.0f * 1000.0f * 1000.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  ASSERT_EQ(testMpiTaskParallel.pre_processing(), true);

  std::vector<float> out_ref;
  if (ProcRank == 0) {
    out_ref = std::vector<float>(count_rows);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, test_run) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (ProcRank == 0) {
    float p1 = -(1000.0f * 1000.0f * 1000.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  ASSERT_EQ(testMpiTaskParallel.run(), true);

  std::vector<float> out_ref;
  if (ProcRank == 0) {
    out_ref = std::vector<float>(count_rows);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    ASSERT_EQ(testMpiTaskSequential.run(), true);
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, test_post_processing) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    float p1 = -(1000.0f * 1000.0f * 1000.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  ASSERT_EQ(testMpiTaskParallel.post_processing(), true);

  std::vector<float> out_ref;
  if (ProcRank == 0) {
    out_ref = std::vector<float>(count_rows);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, matrix18_18) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 18;
  const size_t count_colls = 18;
  float epsilon = 0.0001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    float p1 = -(10.0f * 10.0f * 10.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  std::cout << std::endl;
  std::vector<float> out_ref;
  if (ProcRank == 0) {
    out_ref = std::vector<float>(count_rows);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    float eps = 1e-1f;
    std::vector<bool> results(count_rows);
    std::transform(out.begin(), out.end(), out_ref.begin(), results.begin(),
                   [eps](float a, float b) { return std::abs(a - b) <= eps; });

    ASSERT_EQ(std::all_of(results.begin(), results.end(), [](bool value) { return value; }), true);
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, matrix25_25) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 25;
  const size_t count_colls = 25;
  float epsilon = 0.001f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    float p1 = -(100.0f * 100.0f * 100.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  std::cout << std::endl;
  std::vector<float> out_ref;
  if (ProcRank == 0) {
    out_ref = std::vector<float>(count_rows);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    std::cout << "seq solution" << std::endl;
    for (size_t i = 0; i < count_rows; i++) {
      std::cout << out_ref[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "mpi solution" << std::endl;
    for (size_t i = 0; i < count_rows; i++) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
    float eps = 1e-2f;
    std::vector<bool> results(count_rows);
    std::transform(out.begin(), out.end(), out_ref.begin(), results.begin(),
                   [eps](float a, float b) { return std::abs(a - b) <= eps; });

    ASSERT_EQ(std::all_of(results.begin(), results.end(), [](bool value) { return value; }), true);
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, matrix50_50) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 50;
  const size_t count_colls = 50;
  float epsilon = 0.1f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    float p1 = -(1000.0f * 1000.0f * 1000.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  std::cout << std::endl;
  std::vector<float> out_ref;
  if (ProcRank == 0) {
    out_ref = std::vector<float>(count_rows);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    float eps = 1e-5f;
    std::vector<bool> results(count_rows);
    std::transform(out.begin(), out.end(), out_ref.begin(), results.begin(),
                   [eps](float a, float b) { return std::abs(a - b) <= eps; });

    ASSERT_EQ(std::all_of(results.begin(), results.end(), [](bool value) { return value; }), true);
  }
}

TEST(kholin_k_iterative_methods_Seidel_mpi, matrix100_100) {
  int ProcRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  const size_t count_rows = 100;
  const size_t count_colls = 100;
  float epsilon = 0.01f;
  list_ops::ops_ op = list_ops::METHOD_SEIDEL;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> X0;
  std::vector<float> B;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    float p1 = -(1000.0f * 1000.0f * 1000.0f);
    float p2 = -p1;
    in = kholin_k_iterative_methods_Seidel_mpi::gen_matrix_with_diag_pred(count_rows, count_colls, p1, p2);
    out = std::vector<float>(count_rows);
    X0 = std::vector<float>(count_rows, 0.0f);
    B = kholin_k_iterative_methods_Seidel_mpi::gen_vector(count_rows);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_colls);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(count_rows);
  }

  kholin_k_iterative_methods_Seidel_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, op);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  std::cout << std::endl;
  std::vector<float> out_ref;
  if (ProcRank == 0) {
    out_ref = std::vector<float>(count_rows);
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataSeq->inputs_count.emplace_back(count_rows);
    taskDataSeq->inputs_count.emplace_back(count_colls);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ref.data()));
    taskDataSeq->outputs_count.emplace_back(count_rows);

    // Create Task
    kholin_k_iterative_methods_Seidel_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    float eps = 1e-5f;
    std::vector<bool> results(count_rows);
    std::transform(out.begin(), out.end(), out_ref.begin(), results.begin(),
                   [eps](float a, float b) { return std::abs(a - b) <= eps; });

    ASSERT_EQ(std::all_of(results.begin(), results.end(), [](bool value) { return value; }), true);
  }
}