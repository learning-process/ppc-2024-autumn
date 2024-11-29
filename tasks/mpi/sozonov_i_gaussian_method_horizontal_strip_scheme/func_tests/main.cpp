#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sozonov_i_gaussian_method_horizontal_strip_scheme/include/ops_mpi.hpp"

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_mpi, test_10_unknowns) {
  boost::mpi::communicator world;

  const int count = 10;
  std::vector<double> global_mat(count * (count + 1), 0);
  std::vector<double> global_ans(count, 0);
  std::vector<double> ans(count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < count; ++i) {
      global_mat[i * (count + 1) + i] = 1;
      global_mat[i * (count + 1) + count] = i + 1;
    }
    std::iota(ans.begin(), ans.end(), 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(count, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->inputs_count.emplace_back(count);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_ans, ans);
    ASSERT_EQ(global_ans, ans);
  }
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_mpi, test_50_unknowns) {
  boost::mpi::communicator world;

  const int count = 50;
  std::vector<double> global_mat(count * (count + 1), 0);
  std::vector<double> global_ans(count, 0);
  std::vector<double> ans(count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < count; ++i) {
      global_mat[i * (count + 1) + i] = 1;
      global_mat[i * (count + 1) + count] = i + 1;
    }
    std::iota(ans.begin(), ans.end(), 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(count, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->inputs_count.emplace_back(count);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_ans, ans);
    ASSERT_EQ(global_ans, ans);
  }
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_mpi, test_100_unknowns) {
  boost::mpi::communicator world;

  const int count = 100;
  std::vector<double> global_mat(count * (count + 1), 0);
  std::vector<double> global_ans(count, 0);
  std::vector<double> ans(count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < count; ++i) {
      global_mat[i * (count + 1) + i] = 1;
      global_mat[i * (count + 1) + count] = i + 1;
    }
    std::iota(ans.begin(), ans.end(), 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(count, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->inputs_count.emplace_back(count);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_ans, ans);
    ASSERT_EQ(global_ans, ans);
  }
}

TEST(sozonov_i_gaussian_method_horizontal_strip_scheme_mpi, test_200_unknowns) {
  boost::mpi::communicator world;

  const int count = 200;
  std::vector<double> global_mat(count * (count + 1), 0);
  std::vector<double> global_ans(count, 0);
  std::vector<double> ans(count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (size_t i = 0; i < count; ++i) {
      global_mat[i * (count + 1) + i] = 1;
      global_mat[i * (count + 1) + count] = i + 1;
    }
    std::iota(ans.begin(), ans.end(), 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> reference_ans(count, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataSeq->inputs_count.emplace_back(global_mat.size());
    taskDataSeq->inputs_count.emplace_back(count);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    // Create Task
    sozonov_i_gaussian_method_horizontal_strip_scheme_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_ans, ans);
    ASSERT_EQ(global_ans, ans);
  }
}