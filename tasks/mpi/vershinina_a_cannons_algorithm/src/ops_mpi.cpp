#include "mpi/vershinina_a_cannons_algorithm/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/cartesian_communicator.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <vector>

bool vershinina_a_cannons_algorithm::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  n = taskData->inputs_count[0];

  lhs_.n = n;
  rhs_.n = n;

  res_c.n = n;
  res_.n = n;

  lhs_.read(reinterpret_cast<double*>(taskData->inputs[0]));
  rhs_.read(reinterpret_cast<double*>(taskData->inputs[1]));
  res_c = TMatrix<double>::create(n);
  res_ = TMatrix<double>::create(n);

  lhs_.hshift = 0;
  rhs_.vshift = 0;

  return true;
}

bool vershinina_a_cannons_algorithm::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs.size() == 2 && taskData->inputs_count[0] > 0;
}

bool vershinina_a_cannons_algorithm::TestMPITaskSequential::run() {
  internal_order_test();

  for (int k = 0; k < n; k++) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        res_c.at(i, j) = lhs_.at_h(i, j) * rhs_.at_v(i, j);
      }
    }
    for (int t = 0; t < n; t++) {
      for (int s = 0; s < n; s++) {
        res_.at(t, s) += res_c.at(t, s);
      }
    }
    lhs_.hshift++;
    rhs_.vshift++;
  }
  return true;
}

bool vershinina_a_cannons_algorithm::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(res_.data.begin(), res_.data.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

void copy_mat(double* src, std::vector<double>& dst, int n) { dst.assign(src, src + (n * n)); }

bool vershinina_a_cannons_algorithm::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    n = taskData->inputs_count[0];
    copy_mat(reinterpret_cast<double*>(taskData->inputs[0]), in_.first, n);
    copy_mat(reinterpret_cast<double*>(taskData->inputs[1]), in_.second, n);
  }
  return true;
}

bool vershinina_a_cannons_algorithm::TestMPITaskParallel::validation() {
  internal_order_test();
  return world.rank() != 0 || (taskData->inputs.size() == 2 && taskData->inputs_count[0] > 0);
}

int find_most_close_power_of_2(int x) { return std::floor(std::sqrt(x)); }

std::vector<double> mkpad(const std::vector<double>& in, int n, int padding) {
  std::vector<double> res(padding * padding, 0.);
  for (int i = 0; i < n; i++) {
    std::copy(in.begin() + i * n, in.begin() + (i + 1) * n, res.begin() + i * padding);
  }
  return res;
}
std::vector<double> mkblock(const std::vector<double>& in, int padding, int row, int col, int block) {
  std::vector<double> res(block * block, 0.);
  for (int i = 0; i < block; i++) {
    const int idx = (row * block + i) * padding + (col * block);
    std::copy(in.begin() + idx, in.begin() + idx + block, res.begin() + i * block);
  }
  return res;
}

std::pair<int, int> coords(const boost::mpi::cartesian_communicator& cart, int rank) {
  auto coords = cart.coordinates(rank);
  return {coords[0], coords[1]};
}

bool vershinina_a_cannons_algorithm::TestMPITaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, n, 0);

  auto [lhs_, rhs_] = in_;

  const int power = find_most_close_power_of_2(world.size());
  const int involved = std::pow(power, 2);

  if (world.rank() >= involved) {
    world.split(1);
    return true;
  }

  auto active_comm = world.split(0);

  const int padding = power * ((n + power - 1) / power);
  const int block = padding / power;

  const auto padding2 = padding * padding;
  const auto block2 = block * block;

  if (world.rank() == 0) {
    lhs_ = mkpad(lhs_, n, padding);
    rhs_ = mkpad(rhs_, n, padding);
  }

  boost::mpi::cartesian_communicator cart{active_comm, boost::mpi::cartesian_topology{{power, true}, {power, true}},
                                          false};

  const auto [row, col] = coords(cart, cart.rank());

  auto [left_rank, right_rank] = cart.shifted_ranks(1, 1);
  auto [up_rank, down_rank] = cart.shifted_ranks(0, 1);
  std::vector<double> local_lhs(block2, 0.0);
  std::vector<double> local_rhs(block2, 0.0);
  std::vector<double> local_res(block2, 0.0);

  if (cart.rank() == 0) {
    for (int proc = 0; proc < involved; ++proc) {
      const auto [p_row, p_col] = coords(cart, proc);

      auto lblock = mkblock(lhs_, padding, p_row, p_col, block);
      auto rblock = mkblock(rhs_, padding, p_row, p_col, block);

      if (proc == 0) {
        local_lhs = std::move(lblock);
        local_rhs = std::move(rblock);
      } else {
        cart.send(proc, 0, lblock.data(), block2);
        cart.send(proc, 1, rblock.data(), block2);
      }
    }
  } else {
    cart.recv(0, 0, local_lhs.data(), block2);
    cart.recv(0, 1, local_rhs.data(), block2);
  }

  for (int i = 0; i < row; i++) {
    cart.send(right_rank, 2, local_lhs.data(), block2);
    cart.recv(right_rank, 2, local_lhs.data(), block2);
  }
  for (int i = 0; i < col; i++) {
    cart.send(down_rank, 3, local_rhs.data(), block2);
    cart.recv(down_rank, 3, local_rhs.data(), block2);
  }

  for (int s = 0; s < power; s++) {
    for (int i = 0; i < block; i++) {
      for (int l = 0; l < block; l++) {
        double a_il = local_lhs[i * block + l];
        for (int j = 0; j < block; j++) {
          local_res[i * block + j] += a_il * local_rhs[l * block + j];
        }
      }
    }
    cart.send(right_rank, 4, local_lhs.data(), block2);
    cart.recv(right_rank, 4, local_lhs.data(), block2);
    cart.send(down_rank, 5, local_rhs.data(), block2);
    cart.recv(down_rank, 5, local_rhs.data(), block2);
  }

  if (cart.rank() != 0) {
    cart.send(0, 6, local_res.data(), block2);
  } else {
    res_.resize(padding2, 0.0);

    for (int proc = 0; proc < involved; proc++) {
      if (proc == 0) {
        for (int i = 0; i < block; i++) {
          int dest_index = i * padding;
          std::copy(local_res.begin() + i * block, local_res.begin() + (i + 1) * block, res_.begin() + dest_index);
        }
      } else {
        std::vector<double> buf(block2);
        cart.recv(proc, 6, buf.data(), block2);

        const auto [p_row, p_col] = coords(cart, proc);
        const int begin_row = p_row * block;
        const int begin_col = p_col * block;

        for (int i = 0; i < block; i++) {
          std::copy(buf.begin() + i * block, buf.begin() + (i + 1) * block,
                    res_.begin() + (begin_row + i) * padding + begin_col);
        }
      }
    }

    std::vector<double> overall_res(n * n, 0.0);
    for (int i = 0; i < n; i++) {
      std::copy(res_.begin() + i * padding, res_.begin() + i * padding + n, overall_res.begin() + i * n);
    }
    res_ = std::move(overall_res);
  }
  return true;
}

bool vershinina_a_cannons_algorithm::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* data_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), data_ptr);
  }
  return true;
}