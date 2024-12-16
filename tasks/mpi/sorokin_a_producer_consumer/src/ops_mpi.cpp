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
  double lower_bound = 0;
  double upper_bound = 0;
  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    lower_bound = static_cast<double>(taskData->inputs_count[1]);
    upper_bound = static_cast<double>(taskData->inputs_count[2]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    for (unsigned int proc = 1; proc < count_p_ + 1; proc++) {
      world.send(proc, 0, input_.data() + proc - 1, 1);
    }
  }
  broadcast(world, lower_bound, 0);
  broadcast(world, upper_bound, 0);
  lower_bound_ = lower_bound;
  upper_bound_ = upper_bound;

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
  if (count_p_ == 0) return true;

  const int producer_tag = 1;
  const int consumer_tag = 2;
  const int exit_tag = 0;

  if (world.rank() == 0) {
    std::uniform_real_distribution<double> unif(lower_bound_, upper_bound_);
    std::random_device rand_dev;
    std::mt19937 rand_engine(rand_dev());

    int buffer = -1;
    int rank_data = -1;
    bool has_data = false;
    int remaining_consumers = static_cast<int>(count_p_);
    std::vector<boost::mpi::request> active_requests;

    while (true) {
      int message[2];
      boost::mpi::request recv_req = world.irecv(boost::mpi::any_source, 3, message, 2);
      recv_req.wait();

      int rank = message[0];
      int type = message[1];

      if (type == exit_tag) {
        if (rank > static_cast<int>(count_p_)) {
          remaining_consumers--;
        }
      } else if (type == producer_tag && !has_data) {
        int response[2] = {0, 0};
        active_requests.push_back(world.isend(rank, producer_tag, response, 2));

        int data;
        boost::mpi::request data_recv_req = world.irecv(rank, producer_tag, &data, 1);
        data_recv_req.wait();

        buffer = data;
        rank_data = rank;
        has_data = true;
      } else if (type == consumer_tag && has_data && rank_data == rank - static_cast<int>(count_p_)) {
        int response[2] = {buffer, 0};
        active_requests.push_back(world.isend(rank, consumer_tag, response, 2));
        has_data = false;
      } else if (type == consumer_tag) {
        int response[2] = {buffer, 111};
        active_requests.push_back(world.isend(rank, consumer_tag, response, 2));
      } else if (type == producer_tag) {
        int response[2] = {buffer, 111};
        active_requests.push_back(world.isend(rank, producer_tag, response, 2));
      }

      if (remaining_consumers == 0) {
        break;
      }
    }

    for (unsigned int rank = count_p_ + 1; rank <= count_p_ * 2; ++rank) {
      int result[2];
      boost::mpi::request result_recv_req = world.irecv(rank, 4, result, 2);
      result_recv_req.wait();

      int consumer_index = result[0] - (count_p_ + 1);
      res_[consumer_index] = result[1];
    }

    boost::mpi::wait_all(active_requests.begin(), active_requests.end());
  }

  if (static_cast<unsigned int>(world.rank()) <= count_p_ && world.rank() > 0) {
    int val_s = local_input_[0];
    while (true) {
      int message[2] = {world.rank(), producer_tag};
      boost::mpi::request send_req = world.isend(0, 3, message, 2);
      send_req.wait();

      int response[2];
      boost::mpi::request recv_req = world.irecv(0, producer_tag, response, 2);
      recv_req.wait();

      if (response[1] == 0) break;

      std::uniform_real_distribution<double> unif(lower_bound_, upper_bound_);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(unif(rand_engine))));
    }

    boost::mpi::request send_data_req = world.isend(0, producer_tag, &val_s, 1);
    send_data_req.wait();

    int exit_message[2] = {world.rank(), exit_tag};
    boost::mpi::request send_exit_req = world.isend(0, 3, exit_message, 2);
    send_exit_req.wait();
  }

  if (static_cast<unsigned int>(world.rank()) > count_p_ && static_cast<unsigned int>(world.rank()) <= count_p_ * 2) {
    int val_b = 0;
    while (true) {
      int message[2] = {world.rank(), consumer_tag};
      boost::mpi::request send_req = world.isend(0, 3, message, 2);
      send_req.wait();

      int response[2];
      boost::mpi::request recv_req = world.irecv(0, consumer_tag, response, 2);
      recv_req.wait();

      if (response[1] == 0) {
        val_b = response[0];
        break;
      }

      std::uniform_real_distribution<double> unif(lower_bound_, upper_bound_);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(unif(rand_engine))));
    }

    int exit_message[2] = {world.rank(), exit_tag};
    boost::mpi::request send_exit_req = world.isend(0, 3, exit_message, 2);
    send_exit_req.wait();

    int result[2] = {world.rank(), val_b};
    boost::mpi::request send_result_req = world.isend(0, 4, result, 2);
    send_result_req.wait();
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
