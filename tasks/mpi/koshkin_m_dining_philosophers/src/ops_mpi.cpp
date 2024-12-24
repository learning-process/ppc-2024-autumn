// Copyright 2024 Koshkin Matvey

#include "mpi/koshkin_m_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace koshkin_m_dining_philosophers {
void do_job(boost::mpi::communicator& world, const int& answer, int rank, int a, int b) {
  boost::mpi::request send_req1 = world.isend(rank, a, &answer, b);
  send_req1.wait();
}
}  // namespace koshkin_m_dining_philosophers

bool koshkin_m_dining_philosophers::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  wsize = world.size();
  return true;
}

bool koshkin_m_dining_philosophers::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    bool outputs_count = taskData->outputs_count[0] == 1;
    bool num_procs = world.size() > 2;
    return outputs_count && num_procs;
  }
  return true;
}

bool koshkin_m_dining_philosophers::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int tmp = 0;
  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    int* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
    tmp = input_[0];
    for (int rank = 1; rank <= wsize - 1; ++rank) {
      world.send(rank, 5, &tmp, 1);
    }
    nom = tmp;
  }
  if (world.rank() > 0) {
    int a = 0;
    world.recv(0, 5, &a, 1);
    nom = a;
  }

  if (world.rank() == 0) {
    std::vector<bool> fork(wsize - 1, true);
    auto exit = nom * (wsize - 1);

    while (true) {
      std::vector<int> m(4);
      boost::mpi::request recv_req = world.irecv(boost::mpi::any_source, 3, m.data(), 4);
      recv_req.wait();
      int rank = m[0];
      int wish = m[1];
      int l = m[2];
      int r = m[3];

      if (wish == 3) {
        exit--;
        res_ = (l == nom) ? res_ + l : res_;
      } else if (wish == 2) {
        if (rank == wsize - 1) {
          if (r == 1) {
            const int answer = 2;
            fork[rank - 1] = true;
            koshkin_m_dining_philosophers::do_job(world, answer, rank, 2, 1);
          } else {
            fork[0] = true;
            const int answer = 1;
            koshkin_m_dining_philosophers::do_job(world, answer, rank, 2, 1);
          }
        } else {
          if (r == 1) {
            fork[rank - 1] = true;
            const int answer = 2;
            koshkin_m_dining_philosophers::do_job(world, answer, rank, 2, 1);
          } else {
            fork[rank] = true;
            const int answer = 1;
            koshkin_m_dining_philosophers::do_job(world, answer, rank, 2, 1);
          }
        }
      } else if (wish == 1) {
        if (rank == wsize - 1) {
          if (r == 1) {
            if (!fork[0]) {
              const int answer = 0;
              koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
            } else {
              const int answer = 1;
              koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
              fork[0] = false;
            }
          }
          if (l == 1) {
            if (!fork[rank - 1]) {
              const int answer = 0;

              koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
            } else {
              const int answer = 2;
              koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
              fork[rank - 1] = false;
            }
          }

          if (r + l == 0) {
            if (rank % 2 == 0) {
              if (fork[rank - 1]) {
                const int answer = 2;
                koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
                fork[rank - 1] = false;
              } else {
                const int answer = 0;
                koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
              }
            } else {
              if (fork[0]) {
                const int answer = 1;
                koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
                fork[0] = false;
              } else {
                const int answer = 0;
                koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
              }
            }
          }

        } else {
          if (r == 1) {
            if (!fork[rank]) {
              const int answer = 0;
              koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
            } else {
              const int answer = 1;
              koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
              fork[rank] = false;
            }
          }
          if (l == 1) {
            if (!fork[rank - 1]) {
              const int answer = 0;
              koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
            } else {
              const int answer = 2;
              koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
              fork[rank - 1] = false;
            }
          }
          if (r + l == 0) {
            if (rank % 2 == 0) {
              if (fork[rank - 1]) {
                const int answer = 2;
                koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
                fork[rank - 1] = false;
              } else {
                const int answer = 0;
                koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
              }
            } else {
              if (fork[rank]) {
                const int answer = 1;
                koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
                fork[rank] = false;
              } else {
                const int answer = 0;
                koshkin_m_dining_philosophers::do_job(world, answer, rank, 1, 1);
              }
            }
          }
        }
      }
      if (exit == 0) {
        world.barrier();
        break;
      }
    }
  }

  if (world.rank() > 0) {
    auto [quantity_food, wish_eat, left_hand, right_hand] = std::tuple<int, int, int, int>({0, 0, 0, 0});
    while (quantity_food < nom) {
      const double start = 2;
      const double end = 3;
      std::uniform_real_distribution<double> unif(start, end);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(unif(rand_engine))));
      while (true) {
        if (wish_eat == 0) {
          std::vector<int> m = {world.rank(), 1, left_hand, right_hand};
          boost::mpi::request send_req = world.isend(0, 3, m.data(), 4);
          send_req.wait();
          int a;
          boost::mpi::request recv_req = world.irecv(0, 1, &a, 1);
          recv_req.wait();
          if (a == 1) {
            left_hand = 1;
          }
          if (a == 2) {
            right_hand = 1;
          }
          if (left_hand + right_hand == 2) {
            wish_eat = 1;
          }
        } else {
          std::vector<int> m = {world.rank(), 2, left_hand, right_hand};
          boost::mpi::request send_req = world.isend(0, 3, m.data(), 4);
          send_req.wait();
          int a;
          boost::mpi::request recv_req = world.irecv(0, 2, &a, 1);
          recv_req.wait();
          if (a == 1) {
            left_hand = 0;
          }
          if (a == 2) {
            right_hand = 0;
          }
          if (left_hand + right_hand == 0) {
            wish_eat = 0;
            break;
          }
        }
      }
      quantity_food++;
      std::vector<int> exit_m = {world.rank(), 3, quantity_food, quantity_food};
      boost::mpi::request send_req = world.isend(0, 3, exit_m.data(), 4);
      send_req.wait();
    }
    world.barrier();
  }
  return true;
}

bool koshkin_m_dining_philosophers::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}