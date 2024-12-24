#include <gtest/gtest.h>

#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kholin_k_multidimensional_integrals_rectangle/include/ops_mpi.hpp"

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, test_validation) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  if (ProcRank == 0) {
    // Create data
    std::vector<double> ref_I(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, test_pre_processing) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  ASSERT_EQ(testMpiTaskParallel.pre_processing(), true);

  if (ProcRank == 0) {
    // Create data
    std::vector<double> ref_I(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, test_run) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  ASSERT_EQ(testMpiTaskParallel.run(), true);

  if (ProcRank == 0) {
    // Create data
    std::vector<double> ref_I(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    ASSERT_EQ(testMpiTaskSequential.run(), true);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, test_post_processing) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  ASSERT_EQ(testMpiTaskParallel.post_processing(), true);

  if (ProcRank == 0) {
    // Create data
    std::vector<double> ref_I(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, single_integral_one_var) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.001;
  int n = 10;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, single_integral_two_var) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0, 3.0};
  auto f = [](const std::vector<double> &f_values) { return std::exp(-f_values[0] + f_values[1]); };
  std::vector<double> in_lower_limits{-1};
  std::vector<double> in_upper_limits{5};
  double epsilon = 1e-1;
  int n = 25;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
    // Wrap condition procrank
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, double_integral_two_var) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 2;
  std::vector<double> values{0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] * f_values[0] + f_values[1] * f_values[1]; };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{10, 4};
  double epsilon = 1e-3;
  int n = 50;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, double_integral_one_var) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 2;
  std::vector<double> values{-17.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return 289 + f_values[1] * f_values[1]; };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{10, 4};
  double epsilon = 1e-1;
  int n = 100;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, triple_integral_three_var) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + f_values[1] + f_values[2]; };
  std::vector<double> in_lower_limits{-4, 6, 7};
  std::vector<double> in_upper_limits{4, 13, 8};
  double epsilon = 1e-2;
  int n = 33;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, triple_integral_two_var) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + f_values[2]; };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{12, 20, 2};
  double epsilon = 1e-2;
  int n = 17;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, triple_integral_one_var) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, -10.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + (-10.0); };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{12, 20, 2};
  double epsilon = 1e-2;
  int n = 58;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, triple_integral_one_var_high_accuracy) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 5.0, -10.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + 5.0 + (-10.0); };
  std::vector<double> in_lower_limits{0, 5, -3};
  std::vector<double> in_upper_limits{6, 10, 9};
  double epsilon = 1e-6;
  int n = 90;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, double_integral_two_var_high_accuracy) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 2;
  std::vector<double> values{0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + f_values[1] * f_values[1]; };
  std::vector<double> in_lower_limits{-10, 3};
  std::vector<double> in_upper_limits{5, 4};
  double epsilon = 1e-7;
  int n = 5;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, single_integral_two_var_high_accuracy) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0, 3.0};
  auto f = [](const std::vector<double> &f_values) { return std::exp(-f_values[0] + f_values[1]); };
  std::vector<double> in_lower_limits{-1};
  std::vector<double> in_upper_limits{16};
  double epsilon = 1e-9;
  int n = 34;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
    // Wrap condition procrank
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, single_integral_one_var_high_accuracy) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 1;
  std::vector<double> values{0.0};
  auto f = [](const std::vector<double> &f_values) { return std::sin(f_values[0]); };
  std::vector<double> in_lower_limits{0};
  std::vector<double> in_upper_limits{1};
  double epsilon = 0.00000001;
  int n = 67;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}

TEST(kholin_k_multidimensional_integrals_rectangle_mpi, triple_integral_three_var_high_accuracy) {
  int ProcRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  // Create data
  size_t dim = 3;
  std::vector<double> values{0.0, 0.0, 0.0};
  auto f = [](const std::vector<double> &f_values) { return f_values[0] + f_values[1] + f_values[2]; };
  std::vector<double> in_lower_limits{-4, 6, 7};
  std::vector<double> in_upper_limits{4, 13, 8};
  double epsilon = 1e-8;
  int n = 96;
  std::vector<double> out_I(1, 0.0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, f);
  testMpiTaskParallel.validation();
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  std::vector<double> ref_I(1, 0.0);
  if (ProcRank == 0) {
    // Create data

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, f);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
}