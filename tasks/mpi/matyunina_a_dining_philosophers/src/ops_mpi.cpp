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
  if (world.rank() == 0) {
    std::vector<bool> fork(world.size() - 1, true);
    int exit = nom * (world.size() - 1);
    while (true) {
      int m[4];
      world.recv(boost::mpi::any_source, 3, m, 4);
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
            world.send(rank, 2, &answer, 1);
          } else {
            fork[0] = true;
            int answer = 1;
            world.send(rank, 2, &answer, 1);
          }
        } else {
          if (r == 1) {
            fork[rank - 1] = true;
            int answer = 2;
            world.send(rank, 2, &answer, 1);
          } else {
            fork[rank] = true;
            int answer = 1;
            world.send(rank, 2, &answer, 1);
          }
        }
      } else if (wish == 1) {
        if (rank == world.size() - 1) {
          if (r == 1) {
            if (fork[0] == false) {
              int answer = 0;
              world.send(rank, 1, &answer, 1);
            } else {
              int answer = 1;
              world.send(rank, 1, &answer, 1);
              fork[0] = false;
            }
          }
          if (l == 1) {
            if (fork[rank - 1] == false) {
              int answer = 0;
              world.send(rank, 1, &answer, 1);
            } else {
              int answer = 2;
              world.send(rank, 1, &answer, 1);
              fork[rank - 1] = false;
            }
          }

          if (r + l == 0) {
            if (rank % 2 == 0) {
              if (fork[rank - 1] == true) {
                int answer = 2;
                world.send(rank, 1, &answer, 1);
                fork[rank - 1] = false;
              } else {
                int answer = 0;
                world.send(rank, 1, &answer, 1);
              }
            } else {
              if (fork[0] == true) {
                int answer = 1;
                world.send(rank, 1, &answer, 1);
                fork[0] = false;
              } else {
                int answer = 0;
                world.send(rank, 1, &answer, 1);
              }
            }
          }

        } else {
          if (r == 1) {
            if (fork[rank] == false) {
              int answer = 0;
              world.send(rank, 1, &answer, 1);
            } else {
              int answer = 1;
              world.send(rank, 1, &answer, 1);
              fork[rank] = false;
            }
          }
          if (l == 1) {
            if (fork[rank - 1] == false) {
              int answer = 0;
              world.send(rank, 1, &answer, 1);
            } else {
              int answer = 2;
              world.send(rank, 1, &answer, 1);
              fork[rank - 1] = false;
            }
          }
          if (r + l == 0) {
            if (rank % 2 == 0) {
              if (fork[rank - 1] == true) {
                int answer = 2;
                world.send(rank, 1, &answer, 1);
              }
              fork[rank - 1] = false;
            } else {
              int answer = 0;
              world.send(rank, 1, &answer, 1);
            }
          } else {
            if (fork[rank] == true) {
              int answer = 1;
              world.send(rank, 1, &answer, 1);
              fork[rank] = false;
            } else {
              int answer = 0;
              world.send(rank, 1, &answer, 1);
            }
          }
        }
      }
    }
    if (exit == 0) {
      break;
    }
  }

  if (world.rank() > 0) {
    int c = 0;
    int eat = 0;
    int l = 0;
    int r = 0;
    while (c < nom) {
      const double s = 2;
      const double e = 3;
      std::uniform_real_distribution<double> unif(s, e);
      std::random_device rand_dev;
      std::mt19937 rand_engine(rand_dev());
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(unif(rand_engine))));

      while (true) {
        if (eat == 0) {
          int m[4] = {world.rank(), 1, l, r};
          world.send(0, 3, m, 4);
          int a;
          world.recv(0, 1, &a, 1);
          if (a == 1) {
            l = 1;
          }
          if (a == 2) {
            r = 1;
          }
          if (l + r == 2) {
            eat = 1;
          }
        } else {
          int m[4] = {world.rank(), 2, l, r};
          world.send(0, 3, m, 4);
          int a;
          world.recv(0, 2, &a, 1);
          if (a == 1) {
            l = 0;
          }
          if (a == 2) {
            r = 0;
          }
          if (l + r == 0) {
            eat = 0;
            break;
          }
        }
      }

      c++;
      int exit_m[4] = {world.rank(), 3, c, c};
      world.send(0, 3, exit_m, 4);
    }
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
