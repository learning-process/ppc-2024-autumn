#include "mpi/laganina_e_dejkstras_a/include/ops_mpi.hpp"

#include <queue>

int laganina_e_dejskras_a_mpi::minDistanceVertex(const std::vector<int>& dist, const std::vector<int>& marker) {
  int minvalue = INT_MAX;
  int res = -1;
  for (int i = 0; i < static_cast<int>(dist.size()); ++i) {
    if (marker[i] == 0 && dist[i] <= minvalue) {
      minvalue = dist[i];
      res = i;
    }
  }
  return res;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  v = static_cast<int>(taskData->inputs_count[0]);
  std::vector<int> matrix_row(v * v, 0);
  for (int i = 0; i < v * v; i++) {
    matrix_row[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
  }

  int num_edges = 0;
  for (int i = 0; i < v * v; i++) {
    if (matrix_row[i] != 0) {
      num_edges++;
    }
  }
  row_ptr.resize(v + 1, 0);
  col_ind.resize(num_edges);
  data.resize(num_edges);
  int edge_index = 0;
  for (int i = 0; i < v; i++) {
    row_ptr[i] = edge_index;
    for (int j = 0; j < v; j++) {
      if (matrix_row[i * v + j] != 0) {
        col_ind[edge_index] = j;
        data[edge_index] = matrix_row[i * v + j];
        edge_index++;
      }
    }
  }
  row_ptr[v] = edge_index;
  return true;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  laganina_e_dejskras_a_mpi::TestMPITaskSequential::dijkstra(0, row_ptr, col_ind, data, v, distances);
  return true;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < v; i++) {
    reinterpret_cast<int*>(taskData->outputs[0])[i] = distances[i];
  }
  return true;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  return true;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0;
  }
  return true;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int num_edges;

  if (world.rank() == 0) {
    v = static_cast<int>(taskData->inputs_count[0]);
    std::vector<int> matrix_row(v * v, 0);
    for (int i = 0; i < v * v; i++) {
      matrix_row[i] = reinterpret_cast<int*>(taskData->inputs[0])[i];
    }

    num_edges = 0;
    for (int i = 0; i < v * v; i++) {
      if (matrix_row[i] != 0) {
        num_edges++;
      }
    }
    row_ptr.resize(v + 1, 0);
    col_ind.resize(num_edges);
    data.resize(num_edges);
    int edge_index = 0;
    for (int i = 0; i < v; i++) {
      row_ptr[i] = edge_index;
      for (int j = 0; j < v; j++) {
        if (matrix_row[i * v + j] != 0) {
          col_ind[edge_index] = j;
          data[edge_index] = matrix_row[i * v + j];
          edge_index++;
        }
      }
    }
    row_ptr[v] = edge_index;
  }

  boost::mpi::broadcast(world, num_edges, 0);
  boost::mpi::broadcast(world, v, 0);
  distances.resize(v, INT_MAX);
  row_ptr.resize(v + 1, 0);
  col_ind.resize(num_edges);
  data.resize(num_edges);
  std::vector<int> res;
  std::vector<int> neighbor;
  std::vector<int> weight;
  std::vector<int> local_neighbor;
  std::vector<int> local_weight;
  int max_rank;
  int delta;
  int last;
  int size;

  boost::mpi::broadcast(world, row_ptr.data(), static_cast<int>(row_ptr.size()), 0);
  boost::mpi::broadcast(world, col_ind.data(), static_cast<int>(col_ind.size()), 0);
  boost::mpi::broadcast(world, data.data(), static_cast<int>(data.size()), 0);

  if (world.rank() == 0) {
    laganina_e_dejskras_a_mpi::TestMPITaskSequential::get_children_with_weights(0, row_ptr, col_ind, data, neighbor,
                                                                                weight);
    if (world.size() >= static_cast<int>(neighbor.size())) {
      max_rank = static_cast<int>(neighbor.size()) - 1;
      delta = 1;
      last = 1;
    } else {
      max_rank = world.size() - 1;
      delta = static_cast<int>(neighbor.size()) / world.size();
      last = delta + (static_cast<int>(neighbor.size()) % world.size());
    }
    size = std::max({last, delta});
  }

  boost::mpi::broadcast(world, max_rank, 0);
  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, last, 0);
  boost::mpi::broadcast(world, size, 0);

  if (world.rank() == 0) {
    int rank = 1;
    local_neighbor.resize(delta);
    std::copy(neighbor.begin(), neighbor.begin() + delta, local_neighbor.begin());
    local_weight.resize(delta);
    std::copy(weight.begin(), weight.begin() + delta, local_weight.begin());
    for (int i = delta; rank <= max_rank; i += delta) {
      if (rank == max_rank) {
        world.send(rank, 0, neighbor.data() + i, last);
        world.send(rank, 0, weight.data() + i, last);
        rank++;
        break;
      }
      world.send(rank, 0, neighbor.data() + i, delta);
      world.send(rank, 0, weight.data() + i, delta);
      rank++;
    }
  } else {
    if (world.rank() == max_rank) {
      local_neighbor.resize(last);
      local_weight.resize(last);
      world.recv(0, 0, local_neighbor.data(), last);
      world.recv(0, 0, local_weight.data(), last);
    } else if (world.rank() < max_rank) {
      local_neighbor.resize(delta);
      local_weight.resize(delta);
      world.recv(0, 0, local_neighbor.data(), delta);
      world.recv(0, 0, local_weight.data(), delta);
    }
  }

  if (world.rank() == 0) {
    distances[0] = 0;
  }
  for (int k = 0; k < size; k++) {
    std::vector<int> tmp;
    if (world.rank() == 0) {
      tmp = distances;
    }
    res.resize(v, INT_MAX);
    if ((!local_neighbor.empty()) && (world.rank() <= max_rank)) {
      int vertex = local_neighbor.back();
      local_neighbor.pop_back();
      int bonus = local_weight.back();
      local_weight.pop_back();
      laganina_e_dejskras_a_mpi::TestMPITaskSequential::dijkstra(vertex, row_ptr, col_ind, data, v, res);
      res[vertex] = 0;
      for (int& t : res) {
        if ((abs(t) >= INT_MAX) || (t < 0) || (abs(bonus) >= INT_MAX) || (bonus < 0) ||
            (abs(bonus - t) >= abs(INT_MAX - t))) {
          t = INT_MAX;
        } else {
          t += bonus;
        }
      }
    }
    boost::mpi::reduce(world, res.data(), v, distances.data(), boost::mpi::minimum<int>(), 0);
    if (world.rank() == 0) {
      for (int i = 0; i < v; i++) {
        distances[i] = std::min({distances[i], tmp[i]});
      }
    }
  }

  return true;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < v; i++) {
      reinterpret_cast<int*>(taskData->outputs[0])[i] = distances[i];
    }
  }
  return true;
}
