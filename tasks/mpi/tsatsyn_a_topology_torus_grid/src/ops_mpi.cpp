// Copyright 2023 Nesterov Alexander
#include "mpi/tsatsyn_a_topology_torus_grid/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
using namespace std::chrono_literals;

std::vector<int> tsatsyn_a_topology_torus_grid_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

// bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskSequential::pre_processing() {
//  internal_order_test();
//  // Init vectors
//  input_ = std::vector<int>(taskData->inputs_count[0]);
//  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
//  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
//    input_[i] = tmp_ptr[i];
//  }
//  // Init value for output
//  res = 0;
//  return true;
//}
//
// bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskSequential::validation() {
//  internal_order_test();
//  // Check count elements of output
//  return taskData->outputs_count[0] == 1;
//}
//
// bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskSequential::run() {
//  internal_order_test();
//  if (ops == "+") {
//    res = std::accumulate(input_.begin(), input_.end(), 0);
//  } else if (ops == "-") {
//    res = -std::accumulate(input_.begin(), input_.end(), 0);
//  } else if (ops == "max") {
//    res = *std::max_element(input_.begin(), input_.end());
//  }
//  return true;
//}
//
// bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskSequential::post_processing() {
//  internal_order_test();
//  reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
//  return true;
//}

void mySend(boost::mpi::communicator& world, int source_rank, int dest_rank, int cols, int rows,
            std::map<std::string, int> neighbors, int& inputs) {
  int current_rank = world.rank();
  int source_row_pos, current_row_pos, dest_row_pos, source_col_pos, current_col_pos, dest_col_pos;
  source_row_pos = source_rank / cols;
  current_row_pos = current_rank / cols;
  dest_row_pos = dest_rank / cols;
  source_col_pos = source_rank % cols;
  current_col_pos = current_rank % cols;
  dest_col_pos = dest_rank % cols;

  /*if (world.rank() == 0) {
    std::cout << "source_row_pos: " << source_row_pos << std::endl
              << "dest_row_pos: " << dest_row_pos << std::endl
              << "source_col_pos: " << source_col_pos << std::endl
              << "dest_col_pos: " << dest_col_pos << std::endl;
  }
  std::cout << world.rank() << " current_row_pos: " << current_row_pos << " current_col_pos: " << current_col_pos
            << std::endl;*/
  //                           /
  if (current_col_pos != source_col_pos && current_row_pos != dest_row_pos) {
    // std::cout << world.rank() << " is out1" << std::endl;
    return;
  }

  int delta_row, delta_col;
  delta_row = dest_row_pos - current_row_pos;
  delta_col = dest_col_pos - current_col_pos;

  //
  int middle_row, middle_col;
  middle_row = rows % 2 == 0 ? (rows / 2) - (1) : rows / 2;
  middle_col = cols % 2 == 0 ? (cols / 2) - (1) : cols / 2;

  if (delta_row != 0) {
    if (abs(source_row_pos - dest_row_pos) <= middle_row) {  //
      if (source_row_pos < dest_row_pos) {                   //
        if ((current_row_pos < source_row_pos || current_row_pos > dest_row_pos)) {
          // std::cout << world.rank() << " is out2" << std::endl;
          return;
        }
      } else {  //
        if ((current_row_pos > source_row_pos || current_row_pos < dest_row_pos)) {
          // std::cout << world.rank() << " is out3" << std::endl;
          return;
        }
      }
    } else {                                //
      if (source_row_pos < dest_row_pos) {  //
        if (current_row_pos > source_row_pos && current_row_pos < dest_row_pos) {
          // std::cout << world.rank() << " is out4" << std::endl;
          return;
        }
      } else {  //
        if (current_row_pos < source_row_pos && current_row_pos > dest_row_pos) {
          // std::cout << world.rank() << " is out5" << std::endl;
          return;
        }
      }
    }
  }

  if (abs(source_col_pos - dest_col_pos) <= middle_col) {  //
    if (source_col_pos < dest_col_pos) {                   //
      if ((current_col_pos < source_col_pos || current_col_pos > dest_col_pos)) {
        // std::cout << world.rank() << " is out6" << std::endl;
        return;
      }
    } else {  //
      if ((current_col_pos > source_col_pos || current_col_pos < dest_col_pos)) {
        // std::cout << world.rank() << " is out7" << std::endl;
        return;
      }
    }
  } else {                                //
    if (source_col_pos < dest_col_pos) {  //
      if ((current_col_pos > source_col_pos && current_col_pos < dest_col_pos)) {
        // std::cout << world.rank() << " is out8" << std::endl;
        return;
      }
    } else {  //
      if ((current_col_pos < source_col_pos && current_col_pos > dest_col_pos)) {
        // std::cout << world.rank() << " is out9" << std::endl;
        return;
      }
    }
  }

  int copy;
  if (source_rank == dest_rank || source_rank < 0 || dest_rank > world.size() || source_rank > world.size()) {  //
    // std::cout << "EXIT:REFLEX" << std::endl;
    return;
  } else if (current_rank == dest_rank) {  //
    if (source_col_pos == dest_col_pos) {  //
      if (abs(dest_row_pos - source_row_pos) == 1) {
        if (source_row_pos < dest_row_pos) {
          // std::cout << "wait from up3" << std::endl;
          world.recv(neighbors["up"], 0, copy);
        } else {
          // std::cout << "wait from down3" << std::endl;
          world.recv(neighbors["down"], 0, copy);
        }

      } else if (abs(dest_row_pos - source_row_pos) > (middle_row)) {  //
        if (source_row_pos < dest_row_pos) {
          // std::cout << "wait from down1" << std::endl;
          world.recv(neighbors["down"], 0, copy);
        } else {
          // std::cout << "wait from up1" << std::endl;
          world.recv(neighbors["up"], 0, copy);
        }
      } else {  //
        if (source_row_pos < dest_row_pos) {
          // std::cout << "wait from up2" << std::endl;
          world.recv(neighbors["up"], 0, copy);
        } else {
          // std::cout << "wait from down2" << std::endl;
          world.recv(neighbors["down"], 0, copy);
        }
      }
    } else {  //
      if (abs(dest_col_pos - source_col_pos) == 1) {
        if (source_col_pos < dest_col_pos) {
          // std::cout << "wait from left3" << std::endl;
          world.recv(neighbors["left"], 0, copy);

        } else {
          // std::cout << "wait from right3" << std::endl;
          world.recv(neighbors["right"], 0, copy);
        }

      } else if (abs(dest_col_pos - source_col_pos) > (middle_col)) {  //
        if (source_col_pos < dest_col_pos) {
          // std::cout << "wait from right1" << std::endl;
          world.recv(neighbors["right"], 0, copy);
        } else {
          // std::cout << "wait from left1" << std::endl;
          world.recv(neighbors["left"], 0, copy);
        }
      } else {  //
        if (source_col_pos < dest_col_pos) {
          // std::cout << "wait from left2" << std::endl;
          world.recv(neighbors["left"], 0, copy);
        } else {
          // std::cout << "wait from right2" << std::endl;
          world.recv(neighbors["right"], 0, copy);
        }
      }
    }
    // inputs.insert(inputs.end(), copy.begin(), copy.end());
    inputs = copy;
    return;
  } else if (current_rank == source_rank) {  //
    if (delta_row != 0) {
      if (delta_row < 0) {
        if (abs(dest_row_pos - source_row_pos) == 1) {
          // std::cout << "to up3" << std::endl;
          world.send(neighbors["up"], 0, inputs);
        } else if (abs(delta_row) <= middle_row) {
          // std::cout << "to up1" << std::endl;
          world.send(neighbors["up"], 0, inputs);
        } else {
          // std::cout << "to down1" << std::endl;
          world.send(neighbors["down"], 0, inputs);
        }
      } else {
        if (abs(dest_row_pos - source_row_pos) == 1) {
          // std::cout << "to down3" << std::endl;
          world.send(neighbors["down"], 0, inputs);
        } else if (abs(delta_row) <= middle_row) {
          // std::cout << "to down2" << std::endl;
          world.send(neighbors["down"], 0, inputs);
        } else {
          // std::cout << "to up2" << std::endl;
          world.send(neighbors["up"], 0, inputs);
        }
      }
    } else {
      if (delta_col < 0) {
        if (abs(dest_col_pos - source_col_pos) == 1) {
          // std::cout << "to left3" << std::endl;
          world.send(neighbors["left"], 0, inputs);
        } else if (abs(delta_col) <= middle_col) {
          // std::cout << "to left1" << std::endl;
          world.send(neighbors["left"], 0, inputs);
        } else {
          // std::cout << "to right1" << std::endl;
          world.send(neighbors["right"], 0, inputs);
        }
      } else {
        if (abs(dest_col_pos - source_col_pos) == 1) {
          // std::cout << "to right3" << std::endl;
          world.send(neighbors["right"], 0, inputs);
        } else if (abs(delta_col) <= middle_col) {
          // std::cout << "to right2" << std::endl;
          world.send(neighbors["right"], 0, inputs);
        } else {
          // std::cout << "to left2" << std::endl;
          world.send(neighbors["left"], 0, inputs);
        }
      }
    }
  } else {                                   //
    if (delta_row != 0) {                    //
      if (delta_row < 0) {                   //
        if (abs(delta_row) <= middle_row) {  //
          // std::cout << world.rank() << " is work1" << std::endl;
          world.recv(neighbors["down"], 0, copy);
          world.send(neighbors["up"], 0, copy);
        } else {  //
          // std::cout << world.rank() << " is work2" << std::endl;
          world.recv(neighbors["up"], 0, copy);
          world.send(neighbors["down"], 0, copy);
        }
      } else {                                 //
        if (abs(delta_row) <= (middle_row)) {  //
                                               // std::cout << world.rank() << " is work3" << std::endl;
          world.recv(neighbors["up"], 0, copy);
          world.send(neighbors["down"], 0, copy);
        } else {  //
          // std::cout << world.rank() << " is work4" << std::endl;
          world.recv(neighbors["down"], 0, copy);
          world.send(neighbors["up"], 0, copy);
        }
      }
    } else if (delta_col != 0) {  //                              (if                                            ,
                                  //                 delta_col=0)
      //
      if (current_col_pos == source_col_pos) {
        // std::cout << "eben" << std::endl;
        if (dest_row_pos - source_row_pos < 0) {                   //
          if (abs(dest_row_pos - source_row_pos) <= middle_row) {  //
            world.recv(neighbors["down"], 0, copy);
          } else {  //
            world.recv(neighbors["up"], 0, copy);
          }
        } else {  //
          if (abs(dest_row_pos - source_row_pos) <= middle_row) {
            world.recv(neighbors["up"], 0, copy);
          } else {
            world.recv(neighbors["down"], 0, copy);
          }
        }
      } else {
        if (delta_col < 0) {  //
          if (abs(delta_col) <= middle_col) {
            world.recv(neighbors["right"], 0, copy);
          } else {
            world.recv(neighbors["left"], 0, copy);
          }
        } else {
          if (abs(delta_col) <= middle_col) {
            world.recv(neighbors["left"], 0, copy);
          } else {
            world.recv(neighbors["right"], 0, copy);
          }
        }
      }
      //
      if (delta_col < 0) {  //
        if (abs(delta_col) <= middle_col) {
          world.send(neighbors["left"], 0, copy);
        } else {
          world.send(neighbors["right"], 0, copy);
        }
      } else {
        if (abs(delta_col) <= middle_col) {
          world.send(neighbors["right"], 0, copy);
        } else {
          world.send(neighbors["left"], 0, copy);
        }
      }
    }
  }
  return;
}
void mySend(boost::mpi::communicator& world, int source_rank, int dest_rank, int cols, int rows,
            std::map<std::string, int> neighbors, std::vector<int>& inputs, std::vector<int> inputs_for_send) {
  int current_rank = world.rank();
  int source_row_pos, current_row_pos, dest_row_pos, source_col_pos, current_col_pos, dest_col_pos;
  source_row_pos = source_rank / cols;
  current_row_pos = current_rank / cols;
  dest_row_pos = dest_rank / cols;
  source_col_pos = source_rank % cols;
  current_col_pos = current_rank % cols;
  dest_col_pos = dest_rank % cols;

  //                           /
  if (current_col_pos != source_col_pos && current_row_pos != dest_row_pos) {
    // std::cout << world.rank() << " is out1" << std::endl;
    return;
  }

  int delta_row, delta_col;
  delta_row = dest_row_pos - current_row_pos;
  delta_col = dest_col_pos - current_col_pos;

  //
  int middle_row, middle_col;
  middle_row = rows % 2 == 0 ? (rows / 2) - (1) : rows / 2;
  middle_col = cols % 2 == 0 ? (cols / 2) - (1) : cols / 2;
  // std::cout << delta_row << "+" << delta_col << "+" << middle_col << std::endl;

  if (delta_row != 0) {
    if (abs(source_row_pos - dest_row_pos) <= middle_row) {  //
      if (source_row_pos < dest_row_pos) {                   //
        if ((current_row_pos < source_row_pos || current_row_pos > dest_row_pos)) {
          // std::cout << world.rank() << " is out2" << std::endl;
          return;
        }
      } else {  //
        if ((current_row_pos > source_row_pos || current_row_pos < dest_row_pos)) {
          // std::cout << world.rank() << " is out3" << std::endl;
          return;
        }
      }
    } else {                                //
      if (source_row_pos < dest_row_pos) {  //
        if (current_row_pos > source_row_pos && current_row_pos < dest_row_pos) {
          // std::cout << world.rank() << " is out4" << std::endl;
          return;
        }
      } else {  //
        if (current_row_pos < source_row_pos && current_row_pos > dest_row_pos) {
          // std::cout << world.rank() << " is out5" << std::endl;
          return;
        }
      }
    }
  }

  if (abs(source_col_pos - dest_col_pos) <= middle_col) {  //
    if (source_col_pos < dest_col_pos) {                   //
      if ((current_col_pos < source_col_pos || current_col_pos > dest_col_pos)) {
        // std::cout << world.rank() << " is out6" << std::endl;
        return;
      }
    } else {  //
      if ((current_col_pos > source_col_pos || current_col_pos < dest_col_pos)) {
        // std::cout << world.rank() << " is out7" << std::endl;
        return;
      }
    }
  } else {                                //
    if (source_col_pos < dest_col_pos) {  //
      if ((current_col_pos > source_col_pos && current_col_pos < dest_col_pos)) {
        // std::cout << world.rank() << " is out8" << std::endl;
        return;
      }
    } else {  //
      if ((current_col_pos < source_col_pos && current_col_pos > dest_col_pos)) {
        // std::cout << world.rank() << " is out9" << std::endl;
        return;
      }
    }
  }
  int sizeinput, limit, delta;
  limit = 10000;
  std::vector<int> copy;
  if (source_rank == dest_rank || source_rank < 0 || dest_rank > world.size() || source_rank > world.size()) {  //
    // std::cout << "EXIT:REFLEX" << std::endl;
    return;
  } else if (current_rank == dest_rank) {
    world.recv(source_rank, 0, delta);
    // myBroadcast(world, neighbors, rows, cols, (world.rank() % cols) == 0, world.rank() % cols, world.rank() / cols,
    // delta);
    for (int i = 0; i < delta; i++) {
      copy.clear();
      //
      if (source_col_pos == dest_col_pos) {  //
        if (abs(dest_row_pos - source_row_pos) == 1) {
          if (source_row_pos < dest_row_pos) {
            // std::cout << "wait from up3" << std::endl;
            world.recv(neighbors["up"], 0, copy);
          } else {
            // std::cout << "wait from down3" << std::endl;
            world.recv(neighbors["down"], 0, copy);
          }

        } else if (abs(dest_row_pos - source_row_pos) > (middle_row)) {  //
          if (source_row_pos < dest_row_pos) {
            // std::cout << "wait from down1" << std::endl;
            world.recv(neighbors["down"], 0, copy);
          } else {
            // std::cout << "wait from up1" << std::endl;
            world.recv(neighbors["up"], 0, copy);
          }
        } else {  //
          if (source_row_pos < dest_row_pos) {
            // std::cout << "wait from up2" << std::endl;
            world.recv(neighbors["up"], 0, copy);
          } else {
            // std::cout << "wait from down2" << std::endl;
            world.recv(neighbors["down"], 0, copy);
          }
        }
      } else {  //
        if (abs(dest_col_pos - source_col_pos) == 1) {
          if (source_col_pos < dest_col_pos) {
            // std::cout << "wait from left3" << std::endl;
            world.recv(neighbors["left"], 0, copy);

          } else {
            // std::cout << "wait from right3" << std::endl;
            world.recv(neighbors["right"], 0, copy);
          }

        } else if (abs(dest_col_pos - source_col_pos) > (middle_col)) {  //
          if (source_col_pos < dest_col_pos) {
            // std::cout << "wait from right1" << std::endl;
            world.recv(neighbors["right"], 0, copy);
          } else {
            // std::cout << "wait from left1" << std::endl;
            world.recv(neighbors["left"], 0, copy);
          }
        } else {  //
          if (source_col_pos < dest_col_pos) {
            // std::cout << "wait from left2" << std::endl;
            world.recv(neighbors["left"], 0, copy);
          } else {
            // std::cout << "wait from right2" << std::endl;
            world.recv(neighbors["right"], 0, copy);
          }
        }
      }
      inputs.insert(inputs.end(), copy.begin(), copy.end());
    }
    return;
  } else if (current_rank == source_rank) {  //
    sizeinput = inputs_for_send.size();
    // std::cout << world.rank() << "+" << delta << std::endl;
    delta = sizeinput < limit ? 1 : (sizeinput % limit == 0 ? (sizeinput / limit) : (std::ceil(sizeinput / limit) + 1));
    std::cout << world.rank() << "+" << delta << std::endl;
    for (int proc = 0; proc < world.size(); proc++) {
      if (proc != source_rank) world.send(proc, 0, delta);
    }
    // myBroadcast(world, neighbors, rows, cols, (world.rank() % cols) == 0, world.rank() % cols, world.rank() / cols,
    // delta);
    for (int i = 0; i < delta; i++) {
      copy.clear();
      int size = inputs_for_send.size();
      int endOfCycle = size - limit * (i) >= limit ? limit * (i + 1) : size;
      // std::cout << "endofCycle " << endOfCycle << std::endl;
      // std::cout << "i*limit " << i*limit << std::endl;
      for (int j = i * limit; j < endOfCycle; j++) copy.push_back(inputs_for_send[j]);
      if (delta_row != 0) {
        if (delta_row < 0) {
          if (abs(dest_row_pos - source_row_pos) == 1) {
            // std::cout << "to up3" << std::endl;
            world.send(neighbors["up"], 0, copy);
          } else if (abs(delta_row) <= middle_row) {
            // std::cout << "to up1" << std::endl;
            world.send(neighbors["up"], 0, copy);
          } else {
            // std::cout << "to down1" << std::endl;
            world.send(neighbors["down"], 0, copy);
          }
        } else {
          if (abs(dest_row_pos - source_row_pos) == 1) {
            // std::cout << "to down3" << std::endl;
            world.send(neighbors["down"], 0, copy);
          } else if (abs(delta_row) <= middle_row) {
            // std::cout << "to down2" << std::endl;
            world.send(neighbors["down"], 0, copy);
          } else {
            // std::cout << "to up2" << std::endl;
            world.send(neighbors["up"], 0, copy);
          }
        }
      } else {
        if (delta_col < 0) {
          if (abs(dest_col_pos - source_col_pos) == 1) {
            // std::cout << "to left3" << std::endl;
            world.send(neighbors["left"], 0, copy);
          } else if (abs(delta_col) <= middle_col) {
            // std::cout << "to left1" << std::endl;
            world.send(neighbors["left"], 0, copy);
          } else {
            // std::cout << "to right1" << std::endl;
            world.send(neighbors["right"], 0, copy);
          }
        } else {
          if (abs(dest_col_pos - source_col_pos) == 1) {
            // std::cout << "to right3" << std::endl;
            world.send(neighbors["right"], 0, copy);
          } else if (abs(delta_col) <= middle_col) {
            // std::cout << "to right2" << std::endl;
            world.send(neighbors["right"], 0, copy);
          } else {
            // std::cout << "to left2" << std::endl;
            world.send(neighbors["left"], 0, copy);
          }
        }
      }
    }
  } else {
    world.recv(source_rank, 0, delta);
    // myBroadcast(world, neighbors, rows, cols, (world.rank() % cols) == 0, world.rank() % cols, world.rank() /
    // cols,delta);
    for (int i = 0; i < delta; i++) {
      copy.clear();
      if (delta_row != 0) {                    //
        if (delta_row < 0) {                   //
          if (abs(delta_row) <= middle_row) {  //
                                               // std::cout << world.rank() << " is work1" << std::endl;
            world.recv(neighbors["down"], 0, copy);
            world.send(neighbors["up"], 0, copy);
          } else {  //
            // std::cout << world.rank() << " is work2" << std::endl;
            world.recv(neighbors["up"], 0, copy);
            world.send(neighbors["down"], 0, copy);
          }
        } else {                                 //
          if (abs(delta_row) <= (middle_row)) {  //
            // std::cout << world.rank() << " is work3" << std::endl;
            world.recv(neighbors["up"], 0, copy);
            world.send(neighbors["down"], 0, copy);
          } else {  //
            // std::cout << world.rank() << " is work4" << std::endl;
            world.recv(neighbors["down"], 0, copy);
            world.send(neighbors["up"], 0, copy);
          }
        }
      } else if (delta_col != 0) {  //                              (if                                            ,
                                    //                      delta_col=0)
        //
        if (current_col_pos == source_col_pos) {
          // std::cout << "eben" << std::endl;
          if (dest_row_pos - source_row_pos < 0) {                   //
            if (abs(dest_row_pos - source_row_pos) <= middle_row) {  //
              world.recv(neighbors["down"], 0, copy);
            } else {  //
              world.recv(neighbors["up"], 0, copy);
            }
          } else {  //
            if (abs(dest_row_pos - source_row_pos) <= middle_row) {
              world.recv(neighbors["up"], 0, copy);
            } else {
              world.recv(neighbors["down"], 0, copy);
            }
          }
        } else {
          if (delta_col < 0) {  //
            if (abs(delta_col) <= middle_col) {
              world.recv(neighbors["right"], 0, copy);
            } else {
              world.recv(neighbors["left"], 0, copy);
            }
          } else {
            if (abs(delta_col) <= middle_col) {
              world.recv(neighbors["left"], 0, copy);
            } else {
              world.recv(neighbors["right"], 0, copy);
            }
          }
        }
        //
        if (delta_col < 0) {  //
          if (abs(delta_col) <= middle_col) {
            world.send(neighbors["left"], 0, copy);
          } else {
            world.send(neighbors["right"], 0, copy);
          }
        } else {
          if (abs(delta_col) <= middle_col) {
            world.send(neighbors["right"], 0, copy);
          } else {
            world.send(neighbors["left"], 0, copy);
          }
        }
      }
    }
  }
  return;
}
void myBroadcast(boost::mpi::communicator& world, std::map<std::string, int> neighbors, int rows, int cols,
                 bool is_main_magistralle, int col_pos, int row_pos, std::vector<int>& inputs) {
  //
  int delta;
  if (world.rank() == 0) {
    int sizeinput = inputs.size();
    // std::cout << "sizeinput " << sizeinput << std::endl;
    int limit = 10000;
    delta = sizeinput < limit ? 1 : (sizeinput % limit == 0 ? (sizeinput / limit) : (std::ceil(sizeinput / limit) + 1));
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, delta);
    }
    // std::cout << "delta " << delta << std::endl;

    for (int i = 0; i < delta; i++) {
      std::vector<int> local_input_data;
      int endOfCycle = sizeinput - limit * (i) >= limit ? limit * (i + 1) : sizeinput;
      // std::cout << "limit*delta " << limit * i << std::endl;
      // std::cout << "endofCycle " << endOfCycle << std::endl;

      for (int j = i * limit; j < endOfCycle; j++) local_input_data.push_back(inputs[j]);
      // std::cout << "local " << i << " " << local_input_data.size() << std::endl;
      //

      //         2 2
      if ((rows == 2) && (cols == 2)) {
        // vec, taskData->inputs_count[0]);
        world.send(neighbors["right"], 0, local_input_data);
        world.send(neighbors["down"], 0, local_input_data);
      }
      //         2
      else if ((rows > 2) && (cols == 2)) {
        world.send(neighbors["right"], 0, local_input_data);
        world.send(neighbors["down"], 0, local_input_data);
        world.send(neighbors["up"], 0, local_input_data);
      }  //                    2 1
      else if ((rows == 1) && (cols == 2)) {
        world.send(neighbors["right"], 0, local_input_data);
      }  //
      else if ((cols > 2) && (rows > 2)) {
        world.send(neighbors["left"], 0, local_input_data);
        world.send(neighbors["right"], 0, local_input_data);
        world.send(neighbors["down"], 0, local_input_data);
        world.send(neighbors["up"], 0, local_input_data);
      }  //       2
      else if ((cols > 2) && (rows == 2)) {
        world.send(neighbors["right"], 0, local_input_data);
        world.send(neighbors["down"], 0, local_input_data);
        world.send(neighbors["left"], 0, local_input_data);
      }  //            1 1,Nx1
      else {
        for (auto proc : neighbors) {
          if (proc.second != world.rank()) {
            world.send(proc.second, 0, local_input_data);
          }
        }
      }
    }

  } else {
    world.recv(0, 0, delta);
    // std::cout << "process " << world.rank() << " " << input_data.size() << std::endl;
    std::vector<int> copy;
    //                      (0-          )
    for (int i = 0; i < delta; i++) {
      if (is_main_magistralle) {
        //                                      (   -        )
        if (rows % 2 == 0) {
          if (rows == 2) {
            world.recv(neighbors["up"], 0, copy);
            inputs.insert(inputs.end(), copy.begin(), copy.end());
            if (neighbors["right"] != neighbors["left"]) {
              world.send(neighbors["left"], 0, copy);
              world.send(neighbors["right"], 0, copy);
            } else {
              world.send(neighbors["left"], 0, copy);
            }
          } else {
            if (row_pos < (rows + 1) / 2) {  //                                   (              )
              world.recv(neighbors["up"], 0, copy);
              world.send(neighbors["down"], 0, copy);
            } else if (row_pos > ((rows + 1) / 2) + 1) {  //                                    (              )
              world.recv(neighbors["down"], 0, copy);
              world.send(neighbors["up"], 0, copy);
            } else if (row_pos == (rows + 1) / 2) {
              world.recv(neighbors["up"], 0, copy);
            } else if (row_pos == ((rows + 1) / 2) + 1) {
              world.recv(neighbors["down"], 0, copy);
            }
            inputs.insert(inputs.end(), copy.begin(), copy.end());
            if (neighbors["right"] != neighbors["left"]) {
              world.send(neighbors["left"], 0, copy);
              world.send(neighbors["right"], 0, copy);
            } else {
              world.send(neighbors["left"], 0, copy);
            }
          }
        }
        //                                        (   -        )
        else {
          if (rows != 1) {
            if (row_pos < (rows - 1) / 2) {  //                                   (              )
              world.recv(neighbors["up"], 0, copy);
              world.send(neighbors["down"], 0, copy);
            } else if (row_pos > ((rows - 1) / 2) + 1) {  //                                    (              )
              world.recv(neighbors["down"], 0, copy);
              world.send(neighbors["up"], 0, copy);
            } else if (row_pos == (rows - 1) / 2) {
              world.recv(neighbors["up"], 0, copy);
            } else if (row_pos == ((rows - 1) / 2) + 1) {
              world.recv(neighbors["down"], 0, copy);
            }
            inputs.insert(inputs.end(), copy.begin(), copy.end());
          }
          if (cols != 1) {
            if (neighbors["right"] != neighbors["left"]) {
              world.send(neighbors["left"], 0, copy);
              world.send(neighbors["right"], 0, copy);
            } else {
              world.send(neighbors["left"], 0, copy);
            }
          }
        }
      }
      //
      else {
        //                                   (   -          )
        if (cols % 2 == 0) {
          if (col_pos < (cols + 1) / 2) {
            world.recv(neighbors["left"], 0, copy);
            world.send(neighbors["right"], 0, copy);
          } else if (col_pos > ((cols + 1) / 2) + 1) {
            world.recv(neighbors["right"], 0, copy);
            world.send(neighbors["left"], 0, copy);
          } else if (col_pos == (cols + 1) / 2) {
            world.recv(neighbors["left"], 0, copy);
          } else if (col_pos == (cols + 1) / 2 + 1) {
            world.recv(neighbors["right"], 0, copy);
          }
          inputs.insert(inputs.end(), copy.begin(), copy.end());
        }
        //                                     (   -          )
        else {
          if (col_pos < (cols - 1) / 2) {
            world.recv(neighbors["left"], 0, copy);
            world.send(neighbors["right"], 0, copy);
          } else if (col_pos > ((cols - 1) / 2) + 1) {
            world.recv(neighbors["right"], 0, copy);
            world.send(neighbors["left"], 0, copy);
          } else if (col_pos == (cols - 1) / 2) {
            world.recv(neighbors["left"], 0, copy);
          } else if (col_pos == ((cols - 1) / 2) + 1) {
            world.recv(neighbors["right"], 0, copy);
          }
          inputs.insert(inputs.end(), copy.begin(), copy.end());
        }
      }
    }
  }
}
void myBroadcast(boost::mpi::communicator& world, std::map<std::string, int> neighbors, int rows, int cols,
                 bool is_main_magistralle, int col_pos, int row_pos, int& inputs) {
  //
  // int delta;
  if (world.rank() == 0) {
    // int sizeinput = inputs.size();
    //  std::cout << "sizeinput " << sizeinput << std::endl;
    // int limit = 10000;
    // delta = sizeinput < limit ? 1 : (sizeinput % limit == 0 ? (sizeinput / limit) : (std::ceil(sizeinput / limit) +
    // 1));
    /*for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, delta);
    }*/
    // std::cout << "delta " << delta << std::endl;

    // for (int i = 0; i < delta; i++) {
    // std::vector<int> local_input_data;
    // int endOfCycle = inputs.size() - limit * (i) >= limit ? limit * (i + 1) : inputs.size();
    //  std::cout << "limit*delta " << limit * i << std::endl;
    //  std::cout << "endofCycle " << endOfCycle << std::endl;

    // for (int j = i * limit; j < endOfCycle; j++) local_input_data.push_back(inputs[j]);
    // std::cout << "local " << i << " " << local_input_data.size() << std::endl;
    //

    //         2 2
    if ((rows == 2) && (cols == 2)) {
      // vec, taskData->inputs_count[0]);
      world.send(neighbors["right"], 0, inputs);
      world.send(neighbors["down"], 0, inputs);
    }
    //         2
    else if ((rows > 2) && (cols == 2)) {
      world.send(neighbors["right"], 0, inputs);
      world.send(neighbors["down"], 0, inputs);
      world.send(neighbors["up"], 0, inputs);
    }  //                    2 1
    else if ((rows == 1) && (cols == 2)) {
      world.send(neighbors["right"], 0, inputs);
    }  //
    else if ((cols > 2) && (rows > 2)) {
      world.send(neighbors["left"], 0, inputs);
      world.send(neighbors["right"], 0, inputs);
      world.send(neighbors["down"], 0, inputs);
      world.send(neighbors["up"], 0, inputs);
    }  //       2
    else if ((cols > 2) && (rows == 2)) {
      world.send(neighbors["right"], 0, inputs);
      world.send(neighbors["down"], 0, inputs);
      world.send(neighbors["left"], 0, inputs);
    }  //            1 1,Nx1
    else {
      for (auto proc : neighbors) {
        if (proc.second != world.rank()) {
          world.send(proc.second, 0, inputs);
        }
      }
    }
    //}

  } else {
    // world.recv(0, 0, delta);
    //  std::cout << "process " << world.rank() << " " << input_data.size() << std::endl;
    int copy;
    copy = 0;
    //                      (0-          )
    // for (int i = 0; i < delta; i++) {
    if (is_main_magistralle) {
      //                                      (   -        )
      if (rows % 2 == 0) {
        if (rows == 2) {
          world.recv(neighbors["up"], 0, copy);
          inputs = copy;
          if (neighbors["right"] != neighbors["left"]) {
            world.send(neighbors["left"], 0, copy);
            world.send(neighbors["right"], 0, copy);
          } else {
            world.send(neighbors["left"], 0, copy);
          }
        } else {
          if (row_pos < (rows + 1) / 2) {  //                                   (              )
            world.recv(neighbors["up"], 0, copy);
            world.send(neighbors["down"], 0, copy);
          } else if (row_pos > ((rows + 1) / 2) + 1) {  //                                    (              )
            world.recv(neighbors["down"], 0, copy);
            world.send(neighbors["up"], 0, copy);
          } else if (row_pos == (rows + 1) / 2) {
            world.recv(neighbors["up"], 0, copy);
          } else if (row_pos == ((rows + 1) / 2) + 1) {
            world.recv(neighbors["down"], 0, copy);
          }
          inputs = copy;
          if (neighbors["right"] != neighbors["left"]) {
            world.send(neighbors["left"], 0, copy);
            world.send(neighbors["right"], 0, copy);
          } else {
            world.send(neighbors["left"], 0, copy);
          }
        }
      }
      //                                        (   -        )
      else {
        if (rows != 1) {
          if (row_pos < (rows - 1) / 2) {  //                                   (              )
            world.recv(neighbors["up"], 0, copy);
            world.send(neighbors["down"], 0, copy);
          } else if (row_pos > ((rows - 1) / 2) + 1) {  //                                    (              )
            world.recv(neighbors["down"], 0, copy);
            world.send(neighbors["up"], 0, copy);
          } else if (row_pos == (rows - 1) / 2) {
            world.recv(neighbors["up"], 0, copy);
          } else if (row_pos == ((rows - 1) / 2) + 1) {
            world.recv(neighbors["down"], 0, copy);
          }
          inputs = copy;
        }
        if (cols != 1) {
          if (neighbors["right"] != neighbors["left"]) {
            world.send(neighbors["left"], 0, copy);
            world.send(neighbors["right"], 0, copy);
          } else {
            world.send(neighbors["left"], 0, copy);
          }
        }
      }
    }
    //
    else {
      //                                   (   -          )
      if (cols % 2 == 0) {
        if (col_pos < (cols + 1) / 2) {
          world.recv(neighbors["left"], 0, copy);
          world.send(neighbors["right"], 0, copy);
        } else if (col_pos > ((cols + 1) / 2) + 1) {
          world.recv(neighbors["right"], 0, copy);
          world.send(neighbors["left"], 0, copy);
        } else if (col_pos == (cols + 1) / 2) {
          world.recv(neighbors["left"], 0, copy);
        } else if (col_pos == (cols + 1) / 2 + 1) {
          world.recv(neighbors["right"], 0, copy);
        }
        inputs = copy;
      }
      //                                     (   -          )
      else {
        if (col_pos < (cols - 1) / 2) {
          world.recv(neighbors["left"], 0, copy);
          world.send(neighbors["right"], 0, copy);
        } else if (col_pos > ((cols - 1) / 2) + 1) {
          world.recv(neighbors["right"], 0, copy);
          world.send(neighbors["left"], 0, copy);
        } else if (col_pos == (cols - 1) / 2) {
          world.recv(neighbors["left"], 0, copy);
        } else if (col_pos == ((cols - 1) / 2) + 1) {
          world.recv(neighbors["right"], 0, copy);
        }
        inputs = copy;
      }
    }
    //}
  }
}
int* hasDivisors(int k) {
  int* mas = new int[k];
  for (int i = 0; i < k; i++) {
    mas[i] = -1;
  }
  int j = 0;
  for (int i = 2; i < k; i++) {
    if (k % i == 0) {
      mas[j] = i;
      j++;
    }
  }
  return mas[0] == -1 ? nullptr : mas;
}
bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  //
  std::map<std::string, int> neighbors;
  int rows, cols, row_pos, col_pos;
  bool is_main_magistralle;
  if (world.rank() == 0) {
    input_data.resize(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_data.begin());
    // vec = reinterpret_cast<int*>(taskData->inputs[0]);
    if (hasDivisors(world.size()) == nullptr) {
      cols = world.size();
      rows = 1;
    } else {
      int* mas_copy = hasDivisors(world.size());
      int i = 0;
      while (mas_copy[i] != -1) {
        i++;
      }
      std::random_device dev;
      std::mt19937 gen(dev());
      int randIndex = gen() % (i - 1 + 1) + 1;
      rows = mas_copy[randIndex - 1];
      cols = world.size() / rows;
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, rows);
      world.send(proc, 1, cols);
    }
  } else {
    world.recv(0, 0, rows);
    world.recv(0, 1, cols);
  }
  /*if (world.rank() == 0) {
    std::cout << "col: " << cols << std::endl;
    std::cout << "row: " << rows << std::endl;
  }*/
  // world.barrier();
  row_pos = world.rank() / cols;
  col_pos = world.rank() % cols;
  is_main_magistralle = (col_pos == 0);
  auto toGetNeighbor = [&](int r, int c) -> int {
    int neighbor_rank = r * cols + c;
    return (neighbor_rank < world.size()) ? neighbor_rank : -1;  // -1         ,               (               )
  };

  //
  neighbors["down"] = (row_pos == rows - 1) ? toGetNeighbor(0, col_pos) : toGetNeighbor(row_pos + 1, col_pos);
  neighbors["left"] = (col_pos == 0) ? toGetNeighbor(row_pos, cols - 1) : toGetNeighbor(row_pos, col_pos - 1);
  neighbors["right"] = (col_pos == cols - 1) ? toGetNeighbor(row_pos, 0) : toGetNeighbor(row_pos, col_pos + 1);
  neighbors["up"] = (row_pos == 0) ? toGetNeighbor(rows - 1, col_pos) : toGetNeighbor(row_pos - 1, col_pos);
  /*for (const auto& neighbor : neighbors) {
    std::cout << "Neighbors of " << world.rank() << ": " << neighbor.first << " , " << neighbor.second << std::endl;
    world.barrier();
  }*/
  myBroadcast(world, neighbors, rows, cols, is_main_magistralle, col_pos, row_pos, input_data);
  // delta = sizeinput < limit ? 1 : (sizeinput % limit == 0 ? (sizeinput / limit) : (std::ceil(sizeinput / limit) +
  // 1)); for (int j = 1; j < world.size(); j++) {
  //   mySend(world, 0, j, cols, rows, neighbors, delta);
  // }
  //mySend(world, 0, world.size() - 1, cols, rows, neighbors, input_data, input_data);
  // std::cout << "size of " << world.rank() << " input : " << input_data.size() << std::endl;
  // world.barrier();
  /*if (world.rank() == 0) std::cout << std::endl << std::endl;
  world.barrier();
  mySend(world, world.size() - 1, 0, cols, rows, neighbors, input_data, input_data);
  std::cout << "size of " << world.rank() << " input : " << input_data.size() << std::endl;
  world.barrier();
  if (world.rank() == 0) std::cout << std::endl << std::endl;
  world.barrier();
  mySend(world, 0, world.size() - 1, cols, rows, neighbors, input_data, input_data);
  std::random_device dev;
  std::mt19937 gen(dev());*/

  // int r;
  // r = gen() % (world.size() / 2) + 1;
  // std::cout << "r" << r;
  // std::cout << "delta" << world.rank() << " " << delta << std::endl;
  // for (int i = 0; i < world.size(); i++) {
  // for (int j = 0; j < world.size(); j++) {
  // }
  //}

  // world.barrier();
  if (world.rank() == (world.size() - 1)) {
    res = input_data.size();
    world.send(0, 0, res);
    // std::cout << "RES= " << res << std::endl;
  }
  //mySend(world, world.size() - 1, 0, cols, rows, neighbors, res);


  if (world.rank() == 0) {
    world.recv(world.size() - 1, 0, res);
   // std::cout << "RES2= " << res << std::endl;
  }
  // world.barrier();
  // std::cout << "size of " << world.rank() << " input : " << input_data.size() << std::endl;
  return true;
}

bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  return true;
}

bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}