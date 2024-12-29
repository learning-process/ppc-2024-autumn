#include "mpi/durynichev_d_radix_batcher/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

namespace durynichev_d_radix_batcher_mpi {

void exchange_data(boost::mpi::communicator& comm, int shift, std::vector<double>& data, bool is_sending) {
  int partner_rank = is_sending ? comm.rank() + shift : comm.rank() - shift;
  if (is_sending) {
    comm.send(partner_rank, 0, data);
    comm.recv(partner_rank, 1, data);
  } else {
    std::vector<double> recv_data(data.size());
    comm.recv(partner_rank, 0, recv_data);
    for (size_t iter = 0; iter < data.size(); iter++) {
      if (recv_data[iter] > data[iter]) {
        auto temp = recv_data[iter];
        recv_data[iter] = data[iter];
        data[iter] = temp;
      }
    }
    comm.send(partner_rank, 1, recv_data);
  }
}

std::unordered_set<int> populate_send_workers(int comm_size, int shift) {
  std::unordered_set<int> send_workers;
  for (int worker = 0; worker < comm_size; worker++) {
    if (worker + shift < comm_size && send_workers.find(worker - shift) == send_workers.end()) {
      send_workers.insert(worker);
    }
  }
  return send_workers;
}

void batcher(boost::mpi::communicator& comm, std::vector<double>& data) {
  for (int shift = comm.size() / 2; shift > 0; shift /= 2) {
    std::unordered_set<int> send_workers = populate_send_workers(comm.size(), shift);
    bool should_send = (send_workers.find(comm.rank()) != send_workers.end());
    exchange_data(comm, shift, data, should_send);
  }
}

bool RadixBatcher::validation() {
  internal_order_test();

  if (world.rank() != 0) {
    return true;
  }

  int val_in_size = taskData->inputs_count[1];
  int val_out_size = taskData->outputs_count[0];

  return val_in_size == val_out_size && val_in_size > 0;
}

bool RadixBatcher::pre_processing() {
  internal_order_test();

  arr_size = *reinterpret_cast<int*>(taskData->inputs[0]);
  if (world.rank() == 0) {
    auto* data_ptr_ = reinterpret_cast<double*>(taskData->inputs[1]);
    int data_size = taskData->inputs_count[1];
    input_data.assign(data_ptr_, data_ptr_ + data_size);

    int output_data_size = taskData->outputs_count[0];
    output_data.resize(output_data_size);

    radixSortDouble(input_data.begin(), input_data.begin() + input_data.size() / 2);
    radixSortDouble(input_data.begin() + input_data.size() / 2, input_data.end(), std::greater<>());
  }

  return true;
}

bool RadixBatcher::run() {
  internal_order_test();

  for (workers_count = 1; workers_count * 2 <= world.size(); workers_count *= 2);

  int isWorker = static_cast<int>(world.rank() < arr_size && world.rank() < workers_count);
  boost::mpi::communicator workers = world.split(isWorker);

  if (isWorker == 0) {
    return true;
  }

  int recv_size = arr_size / workers.size();
  std::vector<double> local_data(recv_size);
  boost::mpi::scatter(workers, input_data.data(), local_data.data(), recv_size, 0);

  batcher(workers, local_data);
  radixSortDouble(local_data.begin(), local_data.end());

  boost::mpi::gather(workers, local_data.data(), recv_size, output_data.data(), 0);

  return true;
}

bool RadixBatcher::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* output_data_ptr_ = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(output_data.begin(), output_data.end(), output_data_ptr_);
  }

  return true;
}

}  // namespace durynichev_d_radix_batcher_mpi
