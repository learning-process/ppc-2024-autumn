#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  if (ProcRank == 0) {
    // Create data
    std::vector<double> ref_I(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
  testMpiTaskParallel.validation();
  ASSERT_EQ(testMpiTaskParallel.pre_processing(), true);

  if (ProcRank == 0) {
    // Create data
    std::vector<double> ref_I(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    ASSERT_EQ(testMpiTaskSequential.pre_processing(), true);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    ASSERT_EQ(testMpiTaskSequential.run(), true);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    ASSERT_EQ(testMpiTaskSequential.post_processing(), true);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  // double I = 0.46;
  //  ASSERT_NEAR(out_I[0], I, epsilon);
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  // double I = 54.4;
  /* ASSERT_NEAR(out_I[0], I, epsilon);*/
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
    // Wrap condition procrank
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  // double I = 913.333;
  /*ASSERT_NEAR(out_I[0], I, epsilon);*/
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  // double I = 6026.7;
  /*ASSERT_NEAR(out_I[0], I, epsilon);*/
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  // double I = 952;
  /*ASSERT_NEAR(out_I[0], I, epsilon);*/
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  // double I = 9450;
  /*ASSERT_NEAR(out_I[0], I, epsilon);*/
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
  delete f_object;
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
  std::vector<double> out_I(1, 0.0);
  enum_ops::operations op = enum_ops::MULTISTEP_SCHEME_METHOD_RECTANGLE;
  kholin_k_multidimensional_integrals_rectangle_mpi::Function *f_object =
      new std::function<double(const std::vector<double> &)>(f);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (ProcRank == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&dim));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(values.size());
    taskDataPar->inputs_count.emplace_back(in_lower_limits.size());
    taskDataPar->inputs_count.emplace_back(in_upper_limits.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_I.data()));
    taskDataPar->outputs_count.emplace_back(out_I.size());
  }

  kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar, *f_object);
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
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(f_object));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_lower_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_upper_limits.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataSeq->inputs_count.emplace_back(values.size());
    taskDataSeq->inputs_count.emplace_back(in_lower_limits.size());
    taskDataSeq->inputs_count.emplace_back(in_upper_limits.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_I.data()));
    taskDataSeq->outputs_count.emplace_back(ref_I.size());

    // Create Task
    kholin_k_multidimensional_integrals_rectangle_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq, op);
    testMpiTaskSequential.validation();
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
  }
  // double I = 900;
  /*ASSERT_NEAR(out_I[0], I, epsilon);*/
  if (ProcRank == 0) {
    ASSERT_NEAR(out_I[0], ref_I[0], epsilon);
  }
  delete f_object;  //
}
//
// int main(int argc, char **argv) {
//  boost::mpi::environment env(argc, argv);
//  boost::mpi::communicator world;
//  ::testing::InitGoogleTest(&argc, argv);
//  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
//  if (world.rank() != 0) {
//    delete listeners.Release(listeners.default_result_printer());
//  }
//  return RUN_ALL_TESTS();
//}