// Filatev Vladislav Sum_of_matrix_elements
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/filatev_v_sum_of_matrix_elements/include/ops_mpi.hpp"

TEST(filatev_v_sum_of_matrix_elements_mpi, Test_Sum_10_10_1) {
  boost::mpi::communicator world;
  const int count = 10;
  std::vector<int> out;
  std::vector<std::vector<int>> in;
  
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = std::vector<std::vector<int>>(count, std::vector<int>(count, 1));
    out = std::vector<int>(1, 0);
    for (int i = 0; i < count; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel sumMatrixparallel(taskDataPar, world);
  ASSERT_EQ(sumMatrixparallel.validation(), true);
  sumMatrixparallel.pre_processing();
  sumMatrixparallel.run();
  sumMatrixparallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(100, out[0]);
  }
}

TEST(filatev_v_sum_of_matrix_elements_mpi, Test_Sum_10_10_r) {
  boost::mpi::communicator world;
  const int count = 10;
  std::vector<int> out;
  std::vector<std::vector<int>> in, refIn;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = filatev_v_sum_of_matrix_elements_mpi::getRandomMatrix(count, count);
    refIn = in;
    out = std::vector<int>(1, 0);
    for (int i = 0; i < count; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel sumMatrixParallel(taskDataPar,world);
  ASSERT_EQ(sumMatrixParallel.validation(), true);
  sumMatrixParallel.pre_processing();
  sumMatrixParallel.run();
  sumMatrixParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> refOut;
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> TaskDataSeq = std::make_shared<ppc::core::TaskData>();
    refOut = std::vector<int>(1, 0);
    for (int i = 0; i < count; i++) {
      TaskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(refIn[i].data()));
    }
    TaskDataSeq->inputs_count.emplace_back(count);
    TaskDataSeq->inputs_count.emplace_back(count);
    TaskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refOut.data()));
    TaskDataSeq->outputs_count.emplace_back(1);

    filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq sumMatriSeq(TaskDataSeq);
    ASSERT_EQ(sumMatriSeq.validation(), true);
    sumMatriSeq.pre_processing();
    sumMatriSeq.run();
    sumMatriSeq.post_processing();

    ASSERT_EQ(out[0], refOut[0]);
  }
}

TEST(filatev_v_sum_of_matrix_elements_mpi, Test_Sum_10_20_r) {
  boost::mpi::communicator world;
  const int size_m = 10, size_n = 20;
  std::vector<int> out;
  std::vector<std::vector<int>> in, refIn;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = filatev_v_sum_of_matrix_elements_mpi::getRandomMatrix(size_n, size_m);
    refIn = in;
    out = std::vector<int>(1, 0);
    for (int i = 0; i < size_m; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(size_n);
    taskDataPar->inputs_count.emplace_back(size_m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel sumMatrixParallel(taskDataPar, world);
  ASSERT_EQ(sumMatrixParallel.validation(), true);
  sumMatrixParallel.pre_processing();
  sumMatrixParallel.run();
  sumMatrixParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> refOut;
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> TaskDataSeq = std::make_shared<ppc::core::TaskData>();
    refOut = std::vector<int>(1, 0);
    for (int i = 0; i < size_m; i++) {
      TaskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(refIn[i].data()));
    }
    TaskDataSeq->inputs_count.emplace_back(size_n);
    TaskDataSeq->inputs_count.emplace_back(size_m);
    TaskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refOut.data()));
    TaskDataSeq->outputs_count.emplace_back(1);

    filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq sumMatriSeq(TaskDataSeq);
    ASSERT_EQ(sumMatriSeq.validation(), true);
    sumMatriSeq.pre_processing();
    sumMatriSeq.run();
    sumMatriSeq.post_processing();

    ASSERT_EQ(out[0], refOut[0]);
  }
}

TEST(filatev_v_sum_of_matrix_elements_mpi, Test_Sum_20_10_r) {
  boost::mpi::communicator world;
  const int size_m = 20, size_n = 10;
  std::vector<int> out;
  std::vector<std::vector<int>> in, refIn;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = filatev_v_sum_of_matrix_elements_mpi::getRandomMatrix(size_n, size_m);
    refIn = in;
    out = std::vector<int>(1, 0);
    for (int i = 0; i < size_m; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(size_n);
    taskDataPar->inputs_count.emplace_back(size_m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel sumMatrixParallel(taskDataPar, world);
  ASSERT_EQ(sumMatrixParallel.validation(), true);
  sumMatrixParallel.pre_processing();
  sumMatrixParallel.run();
  sumMatrixParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> refOut;
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> TaskDataSeq = std::make_shared<ppc::core::TaskData>();
    refOut = std::vector<int>(1, 0);
    for (int i = 0; i < size_m; i++) {
      TaskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(refIn[i].data()));
    }
    TaskDataSeq->inputs_count.emplace_back(size_n);
    TaskDataSeq->inputs_count.emplace_back(size_m);
    TaskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refOut.data()));
    TaskDataSeq->outputs_count.emplace_back(1);

    filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq sumMatriSeq(TaskDataSeq);
    ASSERT_EQ(sumMatriSeq.validation(), true);
    sumMatriSeq.pre_processing();
    sumMatriSeq.run();
    sumMatriSeq.post_processing();

    ASSERT_EQ(out[0], refOut[0]);
  }
}

TEST(filatev_v_sum_of_matrix_elements_mpi, Test_Sum_1_1_r) {
  boost::mpi::communicator world;
  const int size_m = 1, size_n = 1;
  std::vector<int> out;
  std::vector<std::vector<int>> in, refIn;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = filatev_v_sum_of_matrix_elements_mpi::getRandomMatrix(size_n, size_m);
    refIn = in;
    out = std::vector<int>(1, 0);
    for (int i = 0; i < size_m; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(size_n);
    taskDataPar->inputs_count.emplace_back(size_m);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  filatev_v_sum_of_matrix_elements_mpi::SumMatrixParallel sumMatrixParallel(taskDataPar, world);
  ASSERT_EQ(sumMatrixParallel.validation(), true);
  sumMatrixParallel.pre_processing();
  sumMatrixParallel.run();
  sumMatrixParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> refOut;
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> TaskDataSeq = std::make_shared<ppc::core::TaskData>();
    refOut = std::vector<int>(1, 0);
    for (int i = 0; i < size_m; i++) {
      TaskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(refIn[i].data()));
    }
    TaskDataSeq->inputs_count.emplace_back(size_n);
    TaskDataSeq->inputs_count.emplace_back(size_m);
    TaskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refOut.data()));
    TaskDataSeq->outputs_count.emplace_back(1);

    filatev_v_sum_of_matrix_elements_mpi::SumMatrixSeq sumMatriSeq(TaskDataSeq);
    ASSERT_EQ(sumMatriSeq.validation(), true);
    sumMatriSeq.pre_processing();
    sumMatriSeq.run();
    sumMatriSeq.post_processing();

    ASSERT_EQ(out[0], refOut[0]);
  }
}