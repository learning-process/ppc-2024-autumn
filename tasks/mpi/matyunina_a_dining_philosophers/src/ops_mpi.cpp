// Copyright 2023 Nesterov Alexander
#include "mpi/matyunina_a_dining_philosophers/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

bool matyunina_a_dining_philosophers_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  res_ = 0;
  return true;
}

bool matyunina_a_dining_philosophers_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1 && taskData->outputs_count[0] == 1 && world.size() > 2;
  }
  return true;
}

bool matyunina_a_dining_philosophers_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  unsigned int tmp = 0;
  if (world.rank() == 0) {
    input_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    tmp = input_[0];
    for (int rank = 1; rank <= world.size() - 1; ++rank) {
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
    std::vector<bool> fork(world.size() - 1, true);
    int exit = nom * (world.size() - 1);

    while (true) {
      int m[4];
      boost::mpi::request recv_req = world.irecv(boost::mpi::any_source, 3, m, 4);
      recv_req.wait();
      int rank = m[0];
      int wish = m[1];
      int l = m[2];
      int r = m[3];

      if (wish == 3) {
        exit--;
        if (l == nom) res_ += l;
      } else if (wish == 2) {
        if (rank == world.size() - 1) {
          if (r == 1) {
            fork[rank - 1] = true;
            int answer = 2;
            boost::mpi::request send_req1 = world.isend(rank, 2, &answer, 1);
            send_req1.wait();
          } else {
            fork[0] = true;
            int answer = 1;
            boost::mpi::request send_req2 = world.isend(rank, 2, &answer, 1);
            send_req2.wait();
          }
        } else {
          if (r == 1) {
            fork[rank - 1] = true;
            int answer = 2;
            boost::mpi::request send_req3 = world.isend(rank, 2, &answer, 1);
            send_req3.wait();
          } else {
            fork[rank] = true;
            int answer = 1;
            boost::mpi::request send_req4 = world.isend(rank, 2, &answer, 1);
            send_req4.wait();
          }
        }
      } else if (wish == 1) {
        if (rank == world.size() - 1) {
          if (r == 1) {
            if (!fork[0]) {
              int answer = 0;
              boost::mpi::request send_req5 = world.isend(rank, 1, &answer, 1);
              send_req5.wait();
            } else {
              int answer = 1;
              boost::mpi::request send_req6 = world.isend(rank, 1, &answer, 1);
              send_req6.wait();
              fork[0] = false;
            }
          }
          if (l == 1) {
            if (!fork[rank - 1]) {
              int answer = 0;
              boost::mpi::request send_req7 = world.isend(rank, 1, &answer, 1);
              send_req7.wait();
            } else {
              int answer = 2;
              boost::mpi::request send_req8 = world.isend(rank, 1, &answer, 1);
              send_req8.wait();
              fork[rank - 1] = false;
            }
          }

          if (r + l == 0) {
            if (rank % 2 == 0) {
              if (fork[rank - 1]) {
                int answer = 2;
                boost::mpi::request send_req9 = world.isend(rank, 1, &answer, 1);
                send_req9.wait();
                fork[rank - 1] = false;
              } else {
                int answer = 0;
                boost::mpi::request send_req10 = world.isend(rank, 1, &answer, 1);
                send_req10.wait();
              }
            } else {
              if (fork[0]) {
                int answer = 1;
                boost::mpi::request send_req11 = world.isend(rank, 1, &answer, 1);
                send_req11.wait();
                fork[0] = false;
              } else {
                int answer = 0;
                boost::mpi::request send_req12 = world.isend(rank, 1, &answer, 1);
                send_req12.wait();
              }
            }
          }

        } else {
          if (r == 1) {
            if (!fork[rank]) {
              int answer = 0;
              boost::mpi::request send_req13 = world.isend(rank, 1, &answer, 1);
              send_req13.wait();
            } else {
              int answer = 1;
              boost::mpi::request send_req14 = world.isend(rank, 1, &answer, 1);
              send_req14.wait();
              fork[rank] = false;
            }
          }
          if (l == 1) {
            if (!fork[rank - 1]) {
              int answer = 0;
              boost::mpi::request send_req15 = world.isend(rank, 1, &answer, 1);
              send_req15.wait();
            } else {
              int answer = 2;
              boost::mpi::request send_req16 = world.isend(rank, 1, &answer, 1);
              send_req16.wait();
              fork[rank - 1] = false;
            }
          }
          if (r + l == 0) {
            if (rank % 2 == 0) {
              if (fork[rank - 1]) {
                int answer = 2;
                boost::mpi::request send_req17 = world.isend(rank, 1, &answer, 1);
                send_req17.wait();
                fork[rank - 1] = false;
              } else {
                int answer = 0;
                boost::mpi::request send_req18 = world.isend(rank, 1, &answer, 1);
                send_req18.wait();
              }
            } else {
              if (fork[rank]) {
                int answer = 1;
                boost::mpi::request send_req19 = world.isend(rank, 1, &answer, 1);
                send_req19.wait();
                fork[rank] = false;
              } else {
                int answer = 0;
                boost::mpi::request send_req20 = world.isend(rank, 1, &answer, 1);
                send_req20.wait();
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
    int quantity_food = 0;
    int wish_eat = 0;
    int left_hand = 0;
    int right_hand = 0;
    while (quantity_food < nom) {
      const double start = 2;
      const double end = 3;
      std::uniform_real_distribution<double> unif(start, end);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(unif(rand_engine))));
      while (true) {
        if (wish_eat == 0) {
          int m[4] = {world.rank(), 1, left_hand, right_hand};
          boost::mpi::request send_req = world.isend(0, 3, m, 4);
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
          int m[4] = {world.rank(), 2, left_hand, right_hand};
          boost::mpi::request send_req = world.isend(0, 3, m, 4);
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
      int exit_m[4] = {world.rank(), 3, quantity_food, quantity_food};
      boost::mpi::request send_req = world.isend(0, 3, exit_m, 4);
      send_req.wait();
    }
    world.barrier();
  }
  return true;
}

bool matyunina_a_dining_philosophers_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res_;
  }
  return true;
}
