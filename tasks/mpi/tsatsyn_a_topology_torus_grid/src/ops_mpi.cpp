// Copyright 2023 Nesterov Alexander
#include "mpi/tsatsyn_a_topology_torus_grid/include/ops_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
enum class Directions : std::uint8_t { up, left, down, right };
void mySend(boost::mpi::communicator& world, int source_rank, int dest_rank, int cols, int rows,
            std::map<Directions, int> neighbors, int& inputs) {
  int current_rank = world.rank();
  int source_row_pos;
  int current_row_pos;
  int dest_row_pos;
  int source_col_pos;
  int current_col_pos;
  int dest_col_pos;
  source_row_pos = source_rank / cols;
  current_row_pos = current_rank / cols;
  dest_row_pos = dest_rank / cols;
  source_col_pos = source_rank % cols;
  current_col_pos = current_rank % cols;
  dest_col_pos = dest_rank % cols;

  // disabling unnecessary rows / columns
  if (world.size() == 1) {
    return;
  }
  if (current_col_pos != source_col_pos && current_row_pos != dest_row_pos) {
    return;
  }
  int delta_row;
  int delta_col;
  delta_row = dest_row_pos - current_row_pos;
  delta_col = dest_col_pos - current_col_pos;

  // disabling individual cells that are not related to the transfer
  int middle_row;
  int middle_col;
  middle_row = rows % 2 == 0 ? (rows / 2) - (1) : rows / 2;
  middle_col = cols % 2 == 0 ? (cols / 2) - (1) : cols / 2;

  if (delta_row != 0) {
    if (abs(source_row_pos - dest_row_pos) <= middle_row) {  // straight ahead
      if (source_row_pos < dest_row_pos) {                   // lower
        if ((current_row_pos < source_row_pos || current_row_pos > dest_row_pos)) {
          return;
        }
      } else {  // higher
        if ((current_row_pos > source_row_pos || current_row_pos < dest_row_pos)) {
          return;
        }
      }
    } else {                                // bypassing
      if (source_row_pos < dest_row_pos) {  // lower
        if (current_row_pos > source_row_pos && current_row_pos < dest_row_pos) {
          return;
        }
      } else {  // higher
        if (current_row_pos < source_row_pos && current_row_pos > dest_row_pos) {
          return;
        }
      }
    }
  }

  if (abs(source_col_pos - dest_col_pos) <= middle_col) {  // straight ahead
    if (source_col_pos < dest_col_pos) {                   // to the right
      if ((current_col_pos < source_col_pos || current_col_pos > dest_col_pos)) {
        return;
      }
    } else {  // to the left
      if ((current_col_pos > source_col_pos || current_col_pos < dest_col_pos)) {
        return;
      }
    }
  } else {                                // bypassing
    if (source_col_pos < dest_col_pos) {  // to the right
      if ((current_col_pos > source_col_pos && current_col_pos < dest_col_pos)) {
        return;
      }
    } else {  // to the left
      if ((current_col_pos < source_col_pos && current_col_pos > dest_col_pos)) {
        return;
      }
    }
  }

  // reflexivity
  if (source_rank == dest_rank || source_rank < 0 || dest_rank > world.size() || source_rank > world.size()) {
    return;
  }

  int copy;
  if (current_rank == dest_rank) {         // where will the call for the last one come from
    if (source_col_pos == dest_col_pos) {  // if we are in the same column with the desired process immediately
      if (abs(dest_row_pos - source_row_pos) == 1) {
        source_row_pos < dest_row_pos ? world.recv(neighbors[Directions::up], 0, copy)
                                      : world.recv(neighbors[Directions::down], 0, copy);
      } else if (abs(dest_row_pos - source_row_pos) > (middle_row)) {  // bypassing
        source_row_pos < dest_row_pos ? world.recv(neighbors[Directions::down], 0, copy)
                                      : world.recv(neighbors[Directions::up], 0, copy);
      } else {  // straight ahead
        source_row_pos < dest_row_pos ? world.recv(neighbors[Directions::up], 0, copy)
                                      : world.recv(neighbors[Directions::down], 0, copy);
      }
    } else {  // if not in one column, the same thing but for columns
      if (abs(dest_col_pos - source_col_pos) == 1) {
        source_col_pos < dest_col_pos ? world.recv(neighbors[Directions::left], 0, copy)
                                      : world.recv(neighbors[Directions::right], 0, copy);
      } else if (abs(dest_col_pos - source_col_pos) > (middle_col)) {  // bypassing
        source_col_pos < dest_col_pos ? world.recv(neighbors[Directions::right], 0, copy)
                                      : world.recv(neighbors[Directions::left], 0, copy);

      } else {  // straight ahead
        source_col_pos < dest_col_pos ? world.recv(neighbors[Directions::left], 0, copy)
                                      : world.recv(neighbors[Directions::right], 0, copy);
      }
    }
    inputs = copy;
  } else if (current_rank == source_rank) {  // when the current one is equal to the source
    if (delta_row != 0) {
      if (delta_row < 0) {
        if (abs(dest_row_pos - source_row_pos) == 1) {
          world.send(neighbors[Directions::up], 0, inputs);
        } else if (abs(delta_row) <= middle_row) {
          world.send(neighbors[Directions::up], 0, inputs);
        } else {
          world.send(neighbors[Directions::down], 0, inputs);
        }
      } else {
        if (abs(dest_row_pos - source_row_pos) == 1) {
          world.send(neighbors[Directions::down], 0, inputs);
        } else if (abs(delta_row) <= middle_row) {
          world.send(neighbors[Directions::down], 0, inputs);
        } else {
          world.send(neighbors[Directions::up], 0, inputs);
        }
      }
    } else {
      if (delta_col < 0) {
        if (abs(dest_col_pos - source_col_pos) == 1) {
          world.send(neighbors[Directions::left], 0, inputs);
        } else if (abs(delta_col) <= middle_col) {
          world.send(neighbors[Directions::left], 0, inputs);
        } else {
          world.send(neighbors[Directions::right], 0, inputs);
        }
      } else {
        if (abs(dest_col_pos - source_col_pos) == 1) {
          world.send(neighbors[Directions::right], 0, inputs);
        } else if (abs(delta_col) <= middle_col) {
          world.send(neighbors[Directions::right], 0, inputs);
        } else {
          world.send(neighbors[Directions::left], 0, inputs);
        }
      }
    }
  } else {                                   // when the current one is not the source but also not the dest one
    if (delta_row != 0) {                    // first, get to the desired row
      if (delta_row < 0) {                   // the goal is higher than the current position
        if (abs(delta_row) <= middle_row) {  // straight ahead
          world.recv(neighbors[Directions::down], 0, copy);
          world.send(neighbors[Directions::up], 0, copy);
        } else {  // bypassing
          world.recv(neighbors[Directions::up], 0, copy);
          world.send(neighbors[Directions::down], 0, copy);
        }
      } else {                                 // the goal is below the current position
        if (abs(delta_row) <= (middle_row)) {  // straight ahead
          world.recv(neighbors[Directions::up], 0, copy);
          world.send(neighbors[Directions::down], 0, copy);
        } else {  // bypassing
          world.recv(neighbors[Directions::down], 0, copy);
          world.send(neighbors[Directions::up], 0, copy);
        }
      }
    } else if (delta_col != 0) {
      // now we go to the desired column ("if" here, just to make it clear what we are working with now,
      // delta_col=0 will not get here anyway)
      // we are dealing with where we accept from
      if (current_col_pos == source_col_pos) {
        if (dest_row_pos - source_row_pos < 0) {  // higher
          abs(dest_row_pos - source_row_pos) <= middle_row ? world.recv(neighbors[Directions::down], 0, copy)
                                                           : world.recv(neighbors[Directions::up], 0, copy);

        } else {  // lower
          abs(dest_row_pos - source_row_pos) <= middle_row ? world.recv(neighbors[Directions::up], 0, copy)
                                                           : world.recv(neighbors[Directions::down], 0, copy);
        }
      } else {
        if (delta_col < 0) {  // to the left
          abs(delta_col) <= middle_col ? world.recv(neighbors[Directions::right], 0, copy)
                                       : world.recv(neighbors[Directions::left], 0, copy);

        } else {  // to the right
          abs(delta_col) <= middle_col ? world.recv(neighbors[Directions::left], 0, copy)
                                       : world.recv(neighbors[Directions::right], 0, copy);
        }
      }
      // where are we sending it
      if (delta_col < 0) {  // to the left
        abs(delta_col) <= middle_col ? world.send(neighbors[Directions::left], 0, copy)
                                     : world.send(neighbors[Directions::right], 0, copy);

      } else {  // to the right
        abs(delta_col) <= middle_col ? world.send(neighbors[Directions::right], 0, copy)
                                     : world.send(neighbors[Directions::left], 0, copy);
      }
    }
  }
}
void myBroadcast(boost::mpi::communicator& world, std::map<Directions, int> neighbors, int rows, int cols, int col_pos,
                 int row_pos, std::vector<int>& inputs) {
  int delta;
  bool is_main_magistralle;
  is_main_magistralle = (col_pos == 0);
  if (world.size() == 1) {
    return;
  }
  if (world.rank() == 0) {
    int sizeinput = inputs.size();
    int limit = 10000;
    if (sizeinput < limit) {
      delta = 1;
    } else {
      if (sizeinput % limit == 0) {
        delta = sizeinput / limit;
      } else {
        delta = std::ceil(sizeinput / limit) + 1;
      }
    }
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, delta);
    }

    for (int i = 0; i < delta; i++) {
      std::vector<int> local_input_data;
      int endOfCycle = sizeinput - limit * (i) >= limit ? limit * (i + 1) : sizeinput;
      for (int j = i * limit; j < endOfCycle; j++) local_input_data.push_back(inputs[j]);
      // a set of special cases that defy the template

      // 2x2 square
      if ((rows == 2) && (cols == 2)) {
        world.send(neighbors[Directions::right], 0, local_input_data);
        world.send(neighbors[Directions::down], 0, local_input_data);
      }
      // 2 columns
      else if ((rows > 2) && (cols == 2)) {
        world.send(neighbors[Directions::right], 0, local_input_data);
        world.send(neighbors[Directions::down], 0, local_input_data);
        world.send(neighbors[Directions::up], 0, local_input_data);
      }
      // a special case for 2x1
      else if ((rows == 1) && (cols == 2)) {
        world.send(neighbors[Directions::right], 0, local_input_data);
      } else if ((cols > 2) && (rows > 2)) {
        world.send(neighbors[Directions::left], 0, local_input_data);
        world.send(neighbors[Directions::right], 0, local_input_data);
        world.send(neighbors[Directions::down], 0, local_input_data);
        world.send(neighbors[Directions::up], 0, local_input_data);
      }
      // rows 2
      else if ((cols > 2) && (rows == 2)) {
        world.send(neighbors[Directions::right], 0, local_input_data);
        world.send(neighbors[Directions::down], 0, local_input_data);
        world.send(neighbors[Directions::left], 0, local_input_data);
      }  // template for 1x1,Nx1
      else {
        for (const std::pair<Directions, int> proc : neighbors) {
          if (proc.second != world.rank()) {
            world.send(proc.second, 0, local_input_data);
          }
        }
      }
    }
  } else {
    world.recv(0, 0, delta);
    std::vector<int> copy;
    for (int i = 0; i < delta; i++) {
      // if on the main Magistrale (column 0)
      if (is_main_magistralle) {
        if (rows % 2 == 0) {
          if (rows == 2) {
            world.recv(neighbors[Directions::up], 0, copy);
            inputs.insert(inputs.end(), copy.begin(), copy.end());
            if (neighbors[Directions::right] != neighbors[Directions::left]) {
              world.send(neighbors[Directions::left], 0, copy);
              world.send(neighbors[Directions::right], 0, copy);
            } else {
              world.send(neighbors[Directions::left], 0, copy);
            }
          } else {
            if (row_pos < (rows + 1) / 2) {  // what the left edge should reach (not inclusive)
              world.recv(neighbors[Directions::up], 0, copy);
              world.send(neighbors[Directions::down], 0, copy);
            } else if (row_pos > ((rows + 1) / 2) + 1) {  // what the right edge should reach (not inclusive)
              world.recv(neighbors[Directions::down], 0, copy);
              world.send(neighbors[Directions::up], 0, copy);
            } else if (row_pos == (rows + 1) / 2) {
              world.recv(neighbors[Directions::up], 0, copy);
            } else if (row_pos == ((rows + 1) / 2) + 1) {
              world.recv(neighbors[Directions::down], 0, copy);
            }
            inputs.insert(inputs.end(), copy.begin(), copy.end());
            if (neighbors[Directions::right] != neighbors[Directions::left]) {
              world.send(neighbors[Directions::left], 0, copy);
              world.send(neighbors[Directions::right], 0, copy);
            } else {
              world.send(neighbors[Directions::left], 0, copy);
            }
          }
        } else {
          if (rows != 1) {
            if (row_pos < (rows - 1) / 2) {
              world.recv(neighbors[Directions::up], 0, copy);
              world.send(neighbors[Directions::down], 0, copy);
            } else if (row_pos > ((rows - 1) / 2) + 1) {
              world.recv(neighbors[Directions::down], 0, copy);
              world.send(neighbors[Directions::up], 0, copy);
            } else if (row_pos == (rows - 1) / 2) {
              world.recv(neighbors[Directions::up], 0, copy);
            } else if (row_pos == ((rows - 1) / 2) + 1) {
              world.recv(neighbors[Directions::down], 0, copy);
            }
            inputs.insert(inputs.end(), copy.begin(), copy.end());
          }
          if (cols != 1) {
            if (neighbors[Directions::right] != neighbors[Directions::left]) {
              world.send(neighbors[Directions::left], 0, copy);
              world.send(neighbors[Directions::right], 0, copy);
            } else {
              world.send(neighbors[Directions::left], 0, copy);
            }
          }
        }
      } else {
        if (cols % 2 == 0) {
          if (col_pos < (cols + 1) / 2) {
            world.recv(neighbors[Directions::left], 0, copy);
            world.send(neighbors[Directions::right], 0, copy);
          } else if (col_pos > ((cols + 1) / 2) + 1) {
            world.recv(neighbors[Directions::right], 0, copy);
            world.send(neighbors[Directions::left], 0, copy);
          } else if (col_pos == (cols + 1) / 2) {
            world.recv(neighbors[Directions::left], 0, copy);
          } else if (col_pos == (cols + 1) / 2 + 1) {
            world.recv(neighbors[Directions::right], 0, copy);
          }
          inputs.insert(inputs.end(), copy.begin(), copy.end());
        } else {
          if (col_pos < (cols - 1) / 2) {
            world.recv(neighbors[Directions::left], 0, copy);
            world.send(neighbors[Directions::right], 0, copy);
          } else if (col_pos > ((cols - 1) / 2) + 1) {
            world.recv(neighbors[Directions::right], 0, copy);
            world.send(neighbors[Directions::left], 0, copy);
          } else if (col_pos == (cols - 1) / 2) {
            world.recv(neighbors[Directions::left], 0, copy);
          } else if (col_pos == ((cols - 1) / 2) + 1) {
            world.recv(neighbors[Directions::right], 0, copy);
          }
          inputs.insert(inputs.end(), copy.begin(), copy.end());
        }
      }
    }
  }
}
std::vector<int> hasDivisors(int k) {
  std::vector<int> mas;
  for (int i = 2; i < k; i++) {
    if (k % i == 0) {
      mas.emplace_back(i);
    }
  }
  return mas;
}
bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->outputs_count[0] == 1 && taskData->inputs_count[0] > 0);
  }
  return true;
}
bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_data.resize(taskData->inputs_count[0]);
    auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tempPtr, tempPtr + taskData->inputs_count[0], input_data.begin());
  }
  return true;
}
bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  std::map<Directions, int> neighbors;
  int rows;
  int cols;
  int row_pos;
  int col_pos;
  if (world.rank() == 0) {
    if (hasDivisors(world.size()).empty()) {
      cols = world.size();
      rows = 1;
    } else {
      std::vector<int> mas_copy = hasDivisors(world.size());
      std::random_device dev;
      std::mt19937 gen(dev());
      int randIndex = gen() % (mas_copy.size()) + 1;
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
  row_pos = world.rank() / cols;
  col_pos = world.rank() % cols;
  auto toGetNeighbor = [&](int r, int c) -> int {
    int neighbor_rank = r * cols + c;
    return (neighbor_rank < world.size()) ? neighbor_rank : -1;
  };
  neighbors[Directions::down] = (row_pos == rows - 1) ? toGetNeighbor(0, col_pos) : toGetNeighbor(row_pos + 1, col_pos);
  neighbors[Directions::left] = (col_pos == 0) ? toGetNeighbor(row_pos, cols - 1) : toGetNeighbor(row_pos, col_pos - 1);
  neighbors[Directions::right] =
      (col_pos == cols - 1) ? toGetNeighbor(row_pos, 0) : toGetNeighbor(row_pos, col_pos + 1);
  neighbors[Directions::up] = (row_pos == 0) ? toGetNeighbor(rows - 1, col_pos) : toGetNeighbor(row_pos - 1, col_pos);
  myBroadcast(world, neighbors, rows, cols, col_pos, row_pos, input_data);
  if (world.rank() == (world.size() - 1)) {
    res = input_data.size();
  }
  mySend(world, world.size() - 1, 0, cols, rows, neighbors, res);
  mySend(world, 0, world.size() - 1, cols, rows, neighbors, res);
  mySend(world, world.size() - 1, 0, cols, rows, neighbors, res);
  return true;
}
bool tsatsyn_a_topology_torus_grid_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}