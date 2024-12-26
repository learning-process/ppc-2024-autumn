// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/vladimirova_j_jarvis_method/func_tests/test_val.cpp"
#include "mpi/vladimirova_j_jarvis_method/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, Test_data_10_0) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_10_0;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_10_0.size());
  if (world.rank() == 0) {
    int row = 10;
    int col = 10;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_10_0.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_10_0[0], otv[0]);
  }
}
TEST(Parallel_Operations_MPI, Test_data_10_1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_10_1;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_10_1.size());
  if (world.rank() == 0) {
    int row = 10;
    int col = 10;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_10_1.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_10_1[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_10_2) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_10_2;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_10_2.size());
  if (world.rank() == 0) {
    int row = 10;
    int col = 10;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_10_2.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_10_2[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_5_0) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_5_0;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_5_0.size());
  if (world.rank() == 0) {
    int row = 5;
    int col = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_5_0.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_5_0[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_5_1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_5_1;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_5_1.size());
  if (world.rank() == 0) {
    int row = 5;
    int col = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_5_1.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_5_1[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_5_2) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_5_2;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_5_2.size());
  if (world.rank() == 0) {
    int row = 5;
    int col = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_5_2.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_5_2[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_5_empty) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_5_empty;
  std::vector<int32_t> otv(1);
  if (world.rank() == 0) {
    int row = 5;
    int col = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(Parallel_Operations_MPI, Test_data_one_row) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_5_empty;
  std::vector<int32_t> otv(1);
  if (world.rank() == 0) {
    int row = 1;
    int col = 2;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
TEST(Parallel_Operations_MPI, Test_data_one_col) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_5_empty;
  std::vector<int32_t> otv(1);
  if (world.rank() == 0) {
    int row = 2;
    int col = 1;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(Parallel_Operations_MPI, Test_data_5_one_line_0) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_5_one_line_0;
  std::vector<int32_t> otv(1);
  if (world.rank() == 0) {
    int row = 5;
    int col = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(Parallel_Operations_MPI, Test_data_5_one_line_1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_5_one_line_1;
  std::vector<int32_t> otv(1);
  if (world.rank() == 0) {
    int row = 5;
    int col = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(row);
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(Parallel_Operations_MPI, Test_data_3_full) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_3_full;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_3_full.size());
  if (world.rank() == 0) {
    int row = 3;
    int col = 3;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_3_full.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_3_full[0], otv[0]);
  };
}

TEST(Parallel_Operations_MPI, Test_10_0_s) {
  const int n = 10;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_10_0;
  std::vector<int> ans = vladimirova_j_jarvis_method_mpi::ans_data_10_0;
  std::vector<int> out(ans.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(ans, out);
}

TEST(Parallel_Operations_MPI, Test_10_1_s) {
  const int n = 10;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_10_1;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_10_1.size());
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_10_1[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_10_2_s) {
  const int n = 10;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_10_2;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_10_2.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_10_2[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_5_0_s) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_5_0;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_5_0.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_5_0[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_5_1_s) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_5_1;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_5_1.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_5_1[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_5_2_s) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_5_2;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_5_2.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_5_2[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_data_5_empty_s) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_5_empty;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), false);
}

TEST(Parallel_Operations_MPI, Test_data_one_row_s) {
  const int row = 1;
  const int col = 2;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_5_empty;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), false);
}

TEST(Parallel_Operations_MPI, Test_data_one_col_s) {
  const int row = 5;
  const int col = 1;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_5_empty;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), false);
}

TEST(Parallel_Operations_MPI, Test_data_5_one_line_0_s) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_5_one_line_0;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), false);
}

TEST(Parallel_Operations_MPI, Test_data_5_one_line_1_s) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_5_one_line_1;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), false);
}

TEST(Parallel_Operations_MPI, Test_data_3_full_s) {
  const int n = 3;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_3_full;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_3_full.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_3_full[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_data_10_5_0) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_10_5_0;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_10_5_0.size());
  if (world.rank() == 0) {
    int row = 5;
    int col = 10;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_10_5_0.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_10_5_0[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_10_5_0_s) {
  const int row = 5;
  const int col = 10;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_10_5_0;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_10_5_0.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_10_5_0[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_data_20_0) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_20_0;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_20_0.size());
  if (world.rank() == 0) {
    int row = 20;
    int col = 20;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_20_0.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_20_0[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_20_0_s) {
  const int row = 20;
  const int col = 20;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_20_0;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_20_0.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_20_0[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_data_20_1) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_20_1;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_20_1.size());
  if (world.rank() == 0) {
    int row = 20;
    int col = 20;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_20_1.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_20_1[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_20_1_s) {
  const int row = 20;
  const int col = 20;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_20_1;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_20_1.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_20_1[0], out[0]);
}

TEST(Parallel_Operations_MPI, Test_data_2_5_0) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  // Create TaskData
  std::vector<int> global_vec = vladimirova_j_jarvis_method_mpi::data_2_5_0;
  std::vector<int32_t> otv(vladimirova_j_jarvis_method_mpi::ans_data_2_5_0.size());
  if (world.rank() == 0) {
    int row = 2;
    int col = 5;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(row);
    taskDataPar->inputs_count.emplace_back(col);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(otv.data()));
    taskDataPar->outputs_count.emplace_back(vladimirova_j_jarvis_method_mpi::ans_data_2_5_0.size());
  }

  vladimirova_j_jarvis_method_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();

  testMpiTaskParallel.run();

  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_2_5_0[0], otv[0]);
  }
}

TEST(Parallel_Operations_MPI, Test_data_2_5_s) {
  const int row = 2;
  const int col = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_mpi::data_2_5_0;
  std::vector<int> out(vladimirova_j_jarvis_method_mpi::ans_data_2_5_0.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ(vladimirova_j_jarvis_method_mpi::ans_data_2_5_0[0], out[0]);
}