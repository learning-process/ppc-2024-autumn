#ifndef _FOX_ALGORITHM_H_
#define _FOX_ALGORITHM_H_

#include <cmath>
#include <numeric>
#include <stdexcept>

#include "../modules/core/task/include/task.hpp"
#include "boost/mpi/collectives.hpp"
#include "boost/mpi/communicator.hpp"
#include "boost/mpi/nonblocking.hpp"
#include "boost/mpi/request.hpp"
#include "matrix_operations.hpp"
#include "mpi/khasanyanov_k_multiply_matrix_fox_algorithm_mpi/include/matrix.hpp"

namespace khasanyanov_k_fox_algorithm {

template <typename DataType = int32_t>
class FoxAlgorithm : public ppc::core::Task {
  enum Color : uint8_t { Used, Unused };
  enum Tag : uint8_t { BlockA, BlockB };

 private:
  boost::mpi::communicator world;

  matrix<DataType> A, B, C;
  matrix<DataType> block_a, block_b, block_c;
  // for idempotency
  matrix<DataType> input_a, input_b;

  static void share_for_row(boost::mpi::communicator &, int, matrix<DataType> &, int, int);

 public:
  explicit FoxAlgorithm(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

  [[nodiscard("processors grid ignored")]] inline static std::vector<int> get_processors_grid(int);
  [[nodiscard("block size ignored")]] inline static size_t calculate_block_size(int, const matrix<DataType> &,
                                                                                const matrix<DataType> &);
  [[nodiscard("block grid ignored")]] static BlockGrid<DataType> get_block_grid(size_t, int, const matrix<DataType> &);
  static void convert_to_matrix(const BlockGrid<DataType> &, int, matrix<DataType> &);
};

template <typename DataType>
bool FoxAlgorithm<DataType>::validation() {
  internal_order_test();

  /*
    taskData
    +---------------------------------------------------------------------------------------------------------------+
    |inputs: {1,2,3,4,5,6,7,8}, {1,2,3,4,5,6,7,8,9,10,11,12} | A, B                                                 |
    |inputs_count: 2, 4, 8, 4, 3, 12                         | A.rows, A.columns, A.size, B.rows, B.columns, B.size |
    |outputs: {}                                             | C                                                    |
    |ouputs_count: 2, 3                                      | C.rows, C.columns, C.size                            |
    +---------------------------------------------------------------------------------------------------------------+
  */
  return world.rank() != 0 ||
         (taskData->inputs.size() == 2 && taskData->inputs_count.size() == 6 &&
          taskData->inputs_count[0] * taskData->inputs_count[1] == taskData->inputs_count[2] &&
          taskData->inputs_count[3] * taskData->inputs_count[4] == taskData->inputs_count[5] &&
          taskData->inputs_count[1] == taskData->inputs_count[3] && !taskData->outputs.empty() &&
          taskData->outputs_count.size() == 3 && taskData->outputs_count[0] == taskData->inputs_count[0] &&
          taskData->outputs_count[1] == taskData->inputs_count[4] &&
          taskData->outputs_count[0] * taskData->outputs_count[1] == taskData->outputs_count[2]);
}

template <typename DataType>
bool FoxAlgorithm<DataType>::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto *a = reinterpret_cast<DataType *>(taskData->inputs[0]);
    auto *b = reinterpret_cast<DataType *>(taskData->inputs[1]);
    input_a.data.assign(a, a + taskData->inputs_count[2]);
    input_b.data.assign(b, b + taskData->inputs_count[5]);
    input_a.rows = taskData->inputs_count[0];
    input_a.columns = taskData->inputs_count[1];
    input_b.rows = taskData->inputs_count[3];
    input_b.columns = taskData->inputs_count[4];
    C = matrix<DataType>(input_a.rows, input_b.columns);
  }
  return true;
}

template <typename DataType>
bool FoxAlgorithm<DataType>::run() {
  internal_order_test();

  int num_processors;
  if (world.rank() == 0) {
    std::vector<int> processors_grid = get_processors_grid(world.size());
    num_processors = processors_grid.size();
  }
  boost::mpi::broadcast(world, num_processors, 0);

  if (world.rank() >= num_processors) {
    world.split(Color::Unused);
    return true;
  }

  boost::mpi::communicator comm = world.split(Color::Used);
  const auto rank = comm.rank();
  const auto size = comm.size();
  //  init step l = 0
  if (rank == 0) {
    auto block_size = calculate_block_size(num_processors, input_a, input_b);
    auto blocks_a = get_block_grid(block_size, num_processors, input_a);
    auto blocks_b = get_block_grid(block_size, num_processors, input_b);
    block_a = blocks_a[0];
    block_b = blocks_b[0];
    block_c = matrix<DataType>{block_a.rows};
    for (size_t i = 1; i < blocks_a.size(); ++i) {
      comm.send(i, Tag::BlockA, blocks_a[i]);
      comm.send(i, Tag::BlockB, blocks_b[i]);
    }
  } else {
    comm.recv(0, Tag::BlockA, block_a);
    comm.recv(0, Tag::BlockB, block_b);
    block_c = matrix<DataType>{block_a.rows};
  }
  int p = sqrt(num_processors);
  B = block_b;
  for (int l = 0; l < p; ++l) {
    A = block_a;
    // share A blocks
    for (int i = 0; i < p; ++i) {
      int share_id = i * p + (i + l) % p;
      share_for_row(comm, share_id, A, p, i);
    }
    // multiply block A & block B
    block_c += MatrixOperations::multiply(A, B);

    // share B blocks
    if (l != p - 1) {
      int recv_id = rank - p;
      int share_id = (rank + p) % size;
      if (recv_id < 0) recv_id = size - abs(recv_id);
      auto request = comm.isend(recv_id, Tag::BlockB, B);
      comm.recv(share_id, Tag::BlockB, B);
      boost::mpi::wait_some(&request, &request + 1);
    }
  }

  // collect C matrix
  BlockGrid<DataType> res_grid;
  boost::mpi::gather(comm, block_c, res_grid, 0);

  if (rank == 0) {
    convert_to_matrix(res_grid, p, C);
  }

  return true;
}

template <typename DataType>
bool FoxAlgorithm<DataType>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(C.begin(), C.end(), reinterpret_cast<DataType *>(taskData->outputs[0]));
  }
  return true;
}

template <typename DataType>
void FoxAlgorithm<DataType>::convert_to_matrix(const BlockGrid<DataType> &grid, int p, matrix<DataType> &mt) {
  if (grid.empty()) return;
  const auto m = mt.rows;
  const auto n = mt.columns;
  const auto block_size = grid.front().rows;
  int num_block;
  int id;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      num_block = (i / block_size) * p + (j / block_size);
      id = (i % block_size) * block_size + (j % block_size);
      mt[i * n + j] = grid[num_block][id];
    }
  }
}

template <typename DataType>
void FoxAlgorithm<DataType>::share_for_row(boost::mpi::communicator &comm, int share_id, matrix<DataType> &block, int p,
                                           int i) {
  if (comm.rank() == share_id) {
    for (int recv_id = i * p; recv_id < i * p + p; ++recv_id) {
      if (recv_id != share_id) {
        comm.send(recv_id, Tag::BlockA, block);
      }
    }
  } else if (comm.rank() >= i * p && comm.rank() < i * p + p) {
    comm.recv(share_id, Tag::BlockA, block);
  }
}

template <typename DataType>
BlockGrid<DataType> FoxAlgorithm<DataType>::get_block_grid(size_t block_size, int num_proc,
                                                           const matrix<DataType> &data) {
  std::vector<matrix<DataType>> res(num_proc, matrix<DataType>{block_size});
  size_t p = sqrt(num_proc);
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j < p; ++j) {
      size_t start_column = std::min(j * block_size, data.columns);
      size_t start_row = std::min(i * block_size, data.rows);
      size_t end_column = std::min(j * block_size + block_size, data.columns);
      size_t end_row = std::min(i * block_size + block_size, data.rows);
      if (start_row >= data.rows || start_column >= data.columns) continue;
      for (size_t k = start_row, id = 0; k < end_row; ++k, id += block_size) {
        std::copy(&data[k * data.columns + start_column], &data[k * data.columns + end_column], &(res[i * p + j][id]));
      }
    }
  }

  return res;
}

template <typename DataType>
inline size_t FoxAlgorithm<DataType>::calculate_block_size(int num_proc, const matrix<DataType> &lhs,
                                                           const matrix<DataType> &rhs) {
  size_t p = sqrt(num_proc);
  size_t n = std::max({lhs.rows, lhs.columns, rhs.rows, rhs.columns});
  while (n % p != 0) ++n;
  return n / p;
}

template <typename DataType>
inline std::vector<int> FoxAlgorithm<DataType>::get_processors_grid(int size) {
  if (size < 1) throw std::invalid_argument("Number of processors must be 1 or more");

  int p = floor(sqrt(size));

  std::vector<int> grid(p * p);
  std::iota(grid.begin(), grid.end(), 0);
  return grid;
}

}  // namespace khasanyanov_k_fox_algorithm

#endif