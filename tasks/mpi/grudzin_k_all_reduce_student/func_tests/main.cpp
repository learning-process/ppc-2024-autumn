#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/grudzin_k_all_reduce_student/include/ops_mpi.hpp"

namespace grudzin_k_all_reduce_student_mpi {
inline std::vector<int> GenMatrix(int colums_temp, int rows_temp) {
  std::vector<int> temp(colums_temp * rows_temp);
  for (int i = 0; i < rows_temp; i++) {
    for (int j = 0; j < colums_temp; j++) {
      temp[i * rows_temp + j] = (rand() % 2001) - 1000;
    }
  }
  return temp;
}
}  // namespace grudzin_k_all_reduce_student_mpi

TEST(grudzin_k_all_reduce_student_mpi, Test_Seq_and_My_Ver_17x23) {
  int rows = 17;
  int colums = 23;

  boost::mpi::communicator world;
  std::vector<int> mas(colums * rows);
  mas = grudzin_k_all_reduce_student_mpi::GenMatrix(colums, rows);
  std::vector<int> parallel_results(colums);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
    taskDataPar->outputs_count.emplace_back(parallel_results.size());
  }
  grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization TestMPITaskMyRealization(taskDataPar);
  ASSERT_EQ(TestMPITaskMyRealization.validation(), true);

  TestMPITaskMyRealization.pre_processing();
  TestMPITaskMyRealization.run();

  TestMPITaskMyRealization.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequntional_results(colums, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(colums);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequntional_results.data()));
    taskDataSeq->outputs_count.emplace_back(sequntional_results.size());
    grudzin_k_all_reduce_student_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallel_results, sequntional_results);
  }
}

TEST(grudzin_k_all_reduce_student_mpi, Test_Seq_and_My_Ver_16x32) {
  int rows = 16;
  int colums = 32;

  boost::mpi::communicator world;
  std::vector<int> mas(colums * rows);
  mas = grudzin_k_all_reduce_student_mpi::GenMatrix(colums, rows);
  std::vector<int> parallel_results(colums);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
    taskDataPar->outputs_count.emplace_back(parallel_results.size());
  }
  grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization TestMPITaskMyRealization(taskDataPar);
  ASSERT_EQ(TestMPITaskMyRealization.validation(), true);

  TestMPITaskMyRealization.pre_processing();
  TestMPITaskMyRealization.run();

  TestMPITaskMyRealization.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequntional_results(colums, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(colums);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequntional_results.data()));
    taskDataSeq->outputs_count.emplace_back(sequntional_results.size());
    grudzin_k_all_reduce_student_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallel_results, sequntional_results);
  }
}

TEST(grudzin_k_all_reduce_student_mpi, Test_Seq_and_My_Ver_2000x2000) {
  int rows = 2000;
  int colums = 2000;

  boost::mpi::communicator world;
  std::vector<int> mas(colums * rows);
  mas = grudzin_k_all_reduce_student_mpi::GenMatrix(colums, rows);
  std::vector<int> parallel_results(colums);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
    taskDataPar->outputs_count.emplace_back(parallel_results.size());
  }
  grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization TestMPITaskMyRealization(taskDataPar);
  ASSERT_EQ(TestMPITaskMyRealization.validation(), true);

  TestMPITaskMyRealization.pre_processing();
  TestMPITaskMyRealization.run();

  TestMPITaskMyRealization.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequntional_results(colums, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(colums);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequntional_results.data()));
    taskDataSeq->outputs_count.emplace_back(sequntional_results.size());
    grudzin_k_all_reduce_student_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallel_results, sequntional_results);
  }
}

TEST(grudzin_k_all_reduce_student_mpi, Test_Seq_and_My_Ver_1337) {
  int rows = 13;
  int colums = 37;

  boost::mpi::communicator world;
  std::vector<int> mas(colums * rows);
  mas = grudzin_k_all_reduce_student_mpi::GenMatrix(colums, rows);
  std::vector<int> parallel_results(colums);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
    taskDataPar->outputs_count.emplace_back(parallel_results.size());
  }
  grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization TestMPITaskMyRealization(taskDataPar);
  ASSERT_EQ(TestMPITaskMyRealization.validation(), true);

  TestMPITaskMyRealization.pre_processing();
  TestMPITaskMyRealization.run();

  TestMPITaskMyRealization.post_processing();
  if (world.rank() == 0) {
    std::vector<int> sequntional_results(colums, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(colums);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequntional_results.data()));
    taskDataSeq->outputs_count.emplace_back(sequntional_results.size());
    grudzin_k_all_reduce_student_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallel_results, sequntional_results);
  }
}

TEST(grudzin_k_all_reduce_student_mpi, Test_Seq_and_My_Ver_20x30) {
  int rows = 20;
  int colums = 30;

  boost::mpi::communicator world;
  std::vector<int> mas(colums * rows);
  mas = grudzin_k_all_reduce_student_mpi::GenMatrix(colums, rows);
  std::vector<int> parallel_results(colums);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(parallel_results.data()));
    taskDataPar->outputs_count.emplace_back(parallel_results.size());
  }
  grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization TestMPITaskMyRealization(taskDataPar);
  ASSERT_EQ(TestMPITaskMyRealization.validation(), true);

  TestMPITaskMyRealization.pre_processing();
  TestMPITaskMyRealization.run();

  TestMPITaskMyRealization.post_processing();
  if (world.rank() == 0) {
    std::vector<int> sequntional_results(colums, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(mas.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(colums);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequntional_results.data()));
    taskDataSeq->outputs_count.emplace_back(sequntional_results.size());
    grudzin_k_all_reduce_student_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(parallel_results, sequntional_results);
  }
}

TEST(grudzin_k_all_reduce_student_mpi, Empty_Val) {
  const int rows = 0;
  const int colums = 0;

  boost::mpi::communicator world;
  std::vector<int> matrix = {};
  std::vector<int> res_(colums, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_.data()));
    taskDataPar->outputs_count.emplace_back(res_.size());
    grudzin_k_all_reduce_student_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
    grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization testMpiTaskMR(taskDataPar);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
    ASSERT_EQ(testMpiTaskMR.validation(), false);
  }
}

TEST(grudzin_k_all_reduce_student_mpi, Forget_One_Size_Val) {
  const int rows = 10;
  const int colums = 10;

  boost::mpi::communicator world;
  std::vector<int> matrix(rows * colums);
  std::vector<int> res_(colums, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    // atata forget about colums
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_.data()));
    taskDataPar->outputs_count.emplace_back(res_.size());
    grudzin_k_all_reduce_student_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
    grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization testMpiTaskMR(taskDataPar);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
    ASSERT_EQ(testMpiTaskMR.validation(), false);
  }
}

TEST(grudzin_k_all_reduce_student_mpi, Ans_NEQ_Col_Val) {
  const int rows = 10;
  const int colums = 10;

  boost::mpi::communicator world;
  std::vector<int> matrix(rows * colums, 0);
  std::vector<int> res_(colums - 1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(colums);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_.data()));
    taskDataPar->outputs_count.emplace_back(res_.size());
    grudzin_k_all_reduce_student_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
    grudzin_k_all_reduce_student_mpi::TestMPITaskMyRealization testMpiTaskMR(taskDataPar);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
    ASSERT_EQ(testMpiTaskMR.validation(), false);
  }
}