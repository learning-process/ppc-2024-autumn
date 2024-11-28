#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "mpi/sadikov_I_gather/include/ops_mpi.h"

namespace sadikov_I_gather_mpi {
std::vector<int> GetRandomData(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = (gen() % 100) - 49;
  }
  return vec;
}
}  // namespace sadikov_I_gather_mpi

TEST(sadikov_I_gather_mpi, checkvalidation) {
  const int columns = 15;
  const int rows = 15;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out(columns, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  in = std::vector<int>(rows * columns, 1);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in_index[0]);
  taskData->inputs_count.emplace_back(in_index[1]);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());
  sadikov_I_gather_mpi::MPITask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
}

TEST(sadikov_I_gather_mpi, checkvalidation2) {
  boost::mpi::communicator world;
  const int columns = 10;
  const int rows = 153;
  const int root = 1;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv(taskData);
  sv.SetRoot(root);
  ASSERT_EQ(sv.validation(), true);
}

TEST(sadikov_I_gather_mpi, check_square_matrix) {
  boost::mpi::communicator world;
  const int columns = 15;
  const int rows = 15;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = sadikov_I_gather_mpi::GetRandomData(columns * rows);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(rows, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_gather_mpi, check_rect_matrix) {
  boost::mpi::communicator world;
  const int columns = 150;
  const int rows = 75;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = std::vector<int>(rows * columns, 1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(rows, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_gather_mpi, check_square_matrix2) {
  boost::mpi::communicator world;
  const int columns = 250;
  const int rows = 250;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = sadikov_I_gather_mpi::GetRandomData(columns * rows);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(rows, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_gather_mpi, check_square_matrix3) {
  boost::mpi::communicator world;
  const int columns = 333;
  const int rows = 333;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = sadikov_I_gather_mpi::GetRandomData(columns * rows);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(rows, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_gather_mpi, check_rect_matrix2) {
  boost::mpi::communicator world;
  const int columns = 872;
  const int rows = 51;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = sadikov_I_gather_mpi::GetRandomData(columns * rows);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(rows, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_gather_mpi, check_rect_matrix3) {
  boost::mpi::communicator world;
  const int columns = 400;
  const int rows = 95;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = sadikov_I_gather_mpi::GetRandomData(rows * columns);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(rows, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_gather_mpi, check_one_row) {
  boost::mpi::communicator world;
  const int columns = 20;
  const int rows = 1;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = sadikov_I_gather_mpi::GetRandomData(columns * rows);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(rows, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_gather_mpi, check_one_element) {
  boost::mpi::communicator world;
  const int columns = 1;
  const int rows = 1;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(columns, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = sadikov_I_gather_mpi::GetRandomData(columns * rows);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(columns, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}

TEST(sadikov_I_gather_mpi, check_empty_matrix) {
  boost::mpi::communicator world;
  const int columns = 0;
  const int rows = 0;
  const int root = 0;
  std::vector<int> in;
  std::vector<int> in_index{rows, columns};
  std::vector<int> out_par(rows, 0);
  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = sadikov_I_gather_mpi::GetRandomData(columns * rows);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in_index[0]);
    taskData->inputs_count.emplace_back(in_index[1]);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskData->outputs_count.emplace_back(out_par.size());
  }
  sadikov_I_gather_mpi::MPITaskParallel sv_par(taskData);
  sv_par.SetRoot(root);
  ASSERT_EQ(sv_par.validation(), true);
  sv_par.pre_processing();
  sv_par.run();
  sv_par.post_processing();
  if (world.rank() == root) {
    std::vector<int> out_seq(rows, 0);
    auto taskData_seq = std::make_shared<ppc::core::TaskData>();
    taskData_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData_seq->inputs_count.emplace_back(in_index[0]);
    taskData_seq->inputs_count.emplace_back(in_index[1]);
    taskData_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_seq.data()));
    taskData_seq->outputs_count.emplace_back(out_seq.size());
    sadikov_I_gather_mpi::MPITask sv_seq(taskData_seq);
    ASSERT_EQ(sv_seq.validation(), true);
    sv_seq.pre_processing();
    sv_seq.run();
    sv_seq.post_processing();
    ASSERT_EQ(out_seq, out_par);
  }
}