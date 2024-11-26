// Copyright 2023 Nesterov Alexander
#include "mpi/sorokin_a_producer_consumer/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool sorokin_a_producer_consumer_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  count_p_ = (world.size() - 1) / 2;

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);

    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    for (unsigned int proc = 1; proc < count_p_ + 1; proc++) {
      world.send(proc, 0, input_.data() + proc - 1, 1);
    }
  }

  if (world.rank() != 0 && static_cast<unsigned int>(world.rank()) <= count_p_) {
    local_input_ = std::vector<int>(1);
    world.recv(0, 0, local_input_.data(), 1);
  }

  res_ = std::vector<int>(count_p_, 0);

  return true;
}

bool sorokin_a_producer_consumer_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return world.size() > 1;
  }
  return true;
}

bool sorokin_a_producer_consumer_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  if (world.size() == 2) return true;
  const int producer_tag = 1;
  const int consumer_tag = 2;
  const int exit_tag = 0;

  if (world.rank() == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    int buffer = -1;
    int rank_data = -1;
    bool has_data = false;
    int remaining_consumers = static_cast<int>(count_p_);

    while (true) {
      int message[2];
      world.recv(boost::mpi::any_source, 3, message, 2);
      int rank = message[0];
      int type = message[1];

      if (type == exit_tag) {
        if (rank > static_cast<int>(count_p_)) {
          remaining_consumers--;
        }
      } else if (type == producer_tag && !has_data) {
        int response[2] = {0, 0};
        world.send(rank, producer_tag, response, 2);
        int data;
        world.recv(rank, producer_tag, &data, 1);
        buffer = data;
        rank_data = rank;
        has_data = true;
      } else if (type == consumer_tag && has_data && rank_data == rank - static_cast<int>(count_p_)) {
        int response[2] = {buffer, 0};
        world.send(rank, consumer_tag, response, 2);
        has_data = false;
      } else if (type == consumer_tag) {
        int response[2] = {buffer, 111};
        world.send(rank, consumer_tag, response, 2);
      } else if (type == producer_tag) {
        int response[2] = {buffer, 111};
        world.send(rank, producer_tag, response, 2);
      }
      if (remaining_consumers == 0) {
        break;
      }
    }
    for (unsigned int rank = count_p_ + 1; rank <= count_p_ * 2; ++rank) {
      int result[2];
      world.recv(rank, 4, result, 2);
      int consumer_index = result[0] - (count_p_ + 1);
      res_[consumer_index] = result[1];
    }
  }

  if (static_cast<unsigned int>(world.rank()) <= count_p_ && world.rank() > 0) {
    int val_s = local_input_[0];
    while (true) {
      int message[2] = {world.rank(), producer_tag};
      world.send(0, 3, message, 2);
      int response[2];
      world.recv(0, producer_tag, response, 2);
      if (response[1] == 0) {
        break;
      }
      const double lower_bound = 2;
      const double upper_bound = 6;
      std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(unif(rand_engine))));
    }

    world.send(0, producer_tag, &val_s, 1);

    int exit_message[2] = {world.rank(), exit_tag};
    world.send(0, 3, exit_message, 2);
  }

  if (static_cast<unsigned int>(world.rank()) > count_p_ && static_cast<unsigned int>(world.rank()) <= count_p_ * 2) {
    int val_b = 0;
    while (true) {
      int message[2] = {world.rank(), consumer_tag};
      world.send(0, 3, message, 2);
      int response[2];
      world.recv(0, consumer_tag, response, 2);
      if (response[1] == 0) {
        val_b = response[0];
        break;
      }
      const double lower_bound = 2;
      const double upper_bound = 6;
      std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(unif(rand_engine))));
    }
    int exit_message[2] = {world.rank(), exit_tag};
    world.send(0, 3, exit_message, 2);

    int result[2] = {world.rank(), val_b};
    world.send(0, 4, result, 2);
  }

  return true;
}

bool sorokin_a_producer_consumer_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
