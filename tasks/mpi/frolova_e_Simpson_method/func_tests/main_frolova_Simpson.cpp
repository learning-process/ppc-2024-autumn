// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/frolova_e_Simpson_method/include/ops_mpi_frolova_Simpson.hpp"

//________DIMENSION:1__________________________

TEST(frolova_e_Simpson_method_mpi, one_dimensional_integral_squaresOfX_test) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {8, 1, 1};
  std::vector<int> values_11 = {4, 1};
  std::vector<double> values_2 = {0.0, 2.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  frolova_e_Simpson_method_mpi::SimpsonmethodParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> res_2(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_11.data()));
    taskDataSeq->inputs_count.emplace_back(values_11.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_2.data()));
    taskDataSeq->outputs_count.emplace_back(res_2.size() * sizeof(double));

    frolova_e_Simpson_method_mpi::SimpsonmethodSequential testTaskSequential(taskDataSeq,
                                                                             frolova_e_Simpson_method_mpi::squaresOfX);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(res_2[0], res[0], 0.1);
  }
}

TEST(frolova_e_Simpson_method_mpi, one_dimensional_integral_cubeOfX_test) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {8, 1, 2};
  std::vector<int> values_11 = {8, 1};
  std::vector<double> values_2 = {0.0, 2.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  frolova_e_Simpson_method_mpi::SimpsonmethodParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> res_2(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_11.data()));
    taskDataSeq->inputs_count.emplace_back(values_11.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_2.data()));
    taskDataSeq->outputs_count.emplace_back(res_2.size() * sizeof(double));

    frolova_e_Simpson_method_mpi::SimpsonmethodSequential testTaskSequential(taskDataSeq,
                                                                             frolova_e_Simpson_method_mpi::cubeOfX);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(res_2[0], res[0], 0.1);
  }
}

//________DIMENSION:2__________________________

TEST(frolova_e_Simpson_method_mpi, two_dimensional_integral_sumOfSquaresOfXandY_test) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {10, 2, 3};
  std::vector<int> values_11 = {10, 2};
  std::vector<double> values_2 = {0.0, 2.0, 0.0, 2.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  frolova_e_Simpson_method_mpi::SimpsonmethodParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> res_2(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_11.data()));
    taskDataSeq->inputs_count.emplace_back(values_11.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_2.data()));
    taskDataSeq->outputs_count.emplace_back(res_2.size() * sizeof(double));

    frolova_e_Simpson_method_mpi::SimpsonmethodSequential testTaskSequential(
        taskDataSeq, frolova_e_Simpson_method_mpi::sumOfSquaresOfXandY);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(res_2[0], res[0], 0.1);
  }
}

TEST(frolova_e_Simpson_method_mpi, two_dimensional_integral_ProductOfXAndY_test) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {10, 2, 4};
  std::vector<int> values_11 = {10, 2};
  std::vector<double> values_2 = {1.0, 4.0, 1.0, 4.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  frolova_e_Simpson_method_mpi::SimpsonmethodParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> res_2(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_11.data()));
    taskDataSeq->inputs_count.emplace_back(values_11.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_2.data()));
    taskDataSeq->outputs_count.emplace_back(res_2.size() * sizeof(double));

    frolova_e_Simpson_method_mpi::SimpsonmethodSequential testTaskSequential(
        taskDataSeq, frolova_e_Simpson_method_mpi::ProductOfXAndY);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(res_2[0], res[0], 0.1);
  }
}

//________DIMENSION:3__________________________

TEST(frolova_e_Simpson_method_mpi, two_dimensional_integral_sumOfSquaresOfXandYandZ_test) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {10, 3, 5};
  std::vector<int> values_11 = {10, 3};
  std::vector<double> values_2 = {0.0, 2.0, 0.0, 2.0, 0.0, 2.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  frolova_e_Simpson_method_mpi::SimpsonmethodParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> res_2(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_11.data()));
    taskDataSeq->inputs_count.emplace_back(values_11.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_2.data()));
    taskDataSeq->outputs_count.emplace_back(res_2.size() * sizeof(double));

    frolova_e_Simpson_method_mpi::SimpsonmethodSequential testTaskSequential(
        taskDataSeq, frolova_e_Simpson_method_mpi::sumOfSquaresOfXandYandZ);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(res_2[0], res[0], 0.1);
  }
}

TEST(frolova_e_Simpson_method_mpi, two_dimensional_integral_ProductOfSquaresOfXandYandZ_test) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {10, 3, 6};
  std::vector<int> values_11 = {10, 3};
  std::vector<double> values_2 = {0.0, 2.0, 0.0, 2.0, 0.0, 2.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));
  }

  frolova_e_Simpson_method_mpi::SimpsonmethodParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> res_2(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_11.data()));
    taskDataSeq->inputs_count.emplace_back(values_11.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_2.data()));
    taskDataSeq->outputs_count.emplace_back(res_2.size() * sizeof(double));

    frolova_e_Simpson_method_mpi::SimpsonmethodSequential testTaskSequential(
        taskDataSeq, frolova_e_Simpson_method_mpi::ProductOfSquaresOfXandYandZ);

    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_NEAR(res_2[0], res[0], 0.1);
  }
}

//____________ASSERT_FALSE_______________________

TEST(frolova_e_Simpson_method_mpi, incorrectNumberOfIntervals_test) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {10, 3, 6};
  std::vector<int> values_11 = {10, 3};
  std::vector<double> values_2 = {0.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));

    frolova_e_Simpson_method_mpi::SimpsonmethodParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(frolova_e_Simpson_method_mpi, NumberOfIntervalsIsNotMultipleOfTheDimension_test) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {10, 3, 6};
  std::vector<int> values_11 = {10, 3};
  std::vector<double> values_2 = {0.0, 1.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size() * sizeof(double));

    frolova_e_Simpson_method_mpi::SimpsonmethodParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}