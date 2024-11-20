#include "mpi/kurakin_m_producer_consumer/include/kurakin_producer_consumer_ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> kurakin_m_producer_consumer_mpi::getRandomVector(int sz, int start = 0, int end = 100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (end - start) + start;
  }
  return vec;
}

int kurakin_m_producer_consumer_mpi::getRandomInt(int start = 0, int end = 100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  int res = gen() % (end - start) + start;
  return res;
}

bool kurakin_m_producer_consumer_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool kurakin_m_producer_consumer_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs.size() == 1 && taskData->inputs_count.size() == 3 && taskData->outputs.size() == 1 &&
           taskData->outputs_count.size() == 1 &&
           (int)(taskData->inputs_count[0]) == (int)(taskData->inputs_count[1]) &&
           (int)(taskData->inputs_count[1]) > 0 && (int)(taskData->inputs_count[1]) < world.size() &&
           (int)(taskData->inputs_count[2]) != 0 && world.size() > 1;
  }
  return true;
}

bool kurakin_m_producer_consumer_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int producer_count;
  res = 0;

  if (world.rank() == 0) {
    producer_count = taskData->inputs_count[1];
  }
  boost::mpi::broadcast(world, producer_count, 0);

  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
      input_[i] = tmp_ptr[i];
    }

    int data_count = input_[0];
    for (int i = 1; i < producer_count; i++) {
      int input = input_[i];
      world.send(i, 0, &input, 1);
    }

    int buffer_size = taskData->inputs_count[2];
    // int producer_count = taskData->inputs_count[1];
    int consumer_count = world.size() - producer_count;

    // buffer and producer with rank 0
    std::queue<int> buffer;
    std::queue<int> producer;
    std::queue<int> producer_exit;
    std::queue<int> consumer;

    int data_0 = 0;
    while (true) {
      if (data_count > 0) {
        data_0 = getRandomInt();
        producer.push(0);
        data_count--;
      } else if (data_count == 0) {
        producer_count--;
        data_count--;
      }
      std::vector<int> message_bw(2);
      world.recv(boost::mpi::any_source, 0, message_bw.data(), 2);
      if (message_bw[1] == 0) {  // exit producer
        producer_count--;
        producer_exit.push(message_bw[0]);
      } else if (message_bw[1] == 1) {  // want put producer
        producer.push(message_bw[0]);
      } else if (message_bw[1] == 2) {  // wand get comsumer
        consumer.push(message_bw[0]);
      }
      while (buffer.size() < (size_t)buffer_size && !producer.empty()) {
        int producer_rank = producer.front();
        producer.pop();
        int data = 0;
        if (producer_rank != 0) {
          world.send(producer_rank, 0, &data, 1);
          world.recv(producer_rank, 0, &data, 1);
        } else {
          data = data_0;
        }
        buffer.push(data);
      }
      while (!buffer.empty() && !consumer.empty()) {
        int consumer_rank = consumer.front();
        consumer.pop();
        int data = buffer.front();
        buffer.pop();
        std::vector<int> message_fw = {data, 2};
        world.send(consumer_rank, 0, message_fw.data(), 2);
      }
      if (producer_count == 0 && buffer.empty() && consumer.size() == (size_t)consumer_count) {
        while (!consumer.empty()) {
          int consumer_rank = consumer.front();
          consumer.pop();
          std::vector<int> message_fw = {0, 0};
          world.send(consumer_rank, 0, message_fw.data(), 2);
        }
        while (!producer_exit.empty()) {
          int producer_rank = producer_exit.front();
          producer_exit.pop();
          int message_fw = 0;
          world.send(producer_rank, 0, &message_fw, 1);
        }
        break;
      }
    }

    reduce(world, 0, res, std::plus(), 0);
  } else if (world.rank() < producer_count) {  // producer
    int data_count;
    world.recv(0, 0, &data_count, 1);

    for (int i = 0; i < data_count; i++) {
      int data = getRandomInt();
      std::vector<int> message_fw = {world.rank(), 1};
      world.send(0, 0, message_fw.data(), 2);

      int get_put;
      world.recv(0, 0, &get_put, 1);
      world.send(0, 0, &data, 1);

      int x = getRandomInt(1, 5);
      std::this_thread::sleep_for(std::chrono::milliseconds(x));
    }

    std::vector<int> message = {world.rank(), 0};
    world.send(0, 0, message.data(), 2);

    int exit;
    world.recv(0, 0, &exit, 1);
    reduce(world, 0, res, std::plus(), 0);
  } else {  // consumer
    int cnt_data = 0;

    while (true) {
      std::vector<int> message_fw = {world.rank(), 2};
      world.send(0, 0, message_fw.data(), 2);

      std::vector<int> message_bw(2);
      world.recv(0, 0, message_bw.data(), 2);

      if (message_bw[1] == 0) {
        break;
      }
      cnt_data++;

      int x = getRandomInt(1, 5);
      std::this_thread::sleep_for(std::chrono::milliseconds(x));
    }

    reduce(world, cnt_data, res, std::plus(), 0);
  }
  return true;
}

bool kurakin_m_producer_consumer_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
