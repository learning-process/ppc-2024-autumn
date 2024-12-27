#include "mpi/laganina_e_dejkstras_a/include/ops_mpi.hpp"
#include <boost/mpi/communicator.hpp>

#include <vector>
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
  v = taskData->inputs_count[0];
  int* ptr = reinterpret_cast<int*>(taskData->inputs[0]);

  int* matrix_row = new int[v * v];
  for (int i = 0; i < v * v; i++) {
    matrix_row[i] = ptr[i];
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
  if (taskData->inputs_count[0] <= 0) {
    return false;
  }
  return true;
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
    if (taskData->inputs_count[0] <= 0) {
      return false;
    }
  }
  return true;
}

bool laganina_e_dejskras_a_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  int num_edges;

  if (world.rank() == 0) {
    v = taskData->inputs_count[0];
    int* ptr = reinterpret_cast<int*>(taskData->inputs[0]);

    int* matrix_row = new int[v * v];
    for (int i = 0; i < v * v; i++) {
      matrix_row[i] = ptr[i];
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
    delete[] matrix_row;
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
    laganina_e_dejskras_a_mpi::TestMPITaskSequential::get_children_with_weights(0, row_ptr, col_ind, data,
                                                                                neighbor, weight);
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
  } 
  else {
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
      for (int& t : res) {
        if (t != INT_MAX) {
          if ((t < 0) || (bonus == INT_MAX))
            t = INT_MAX;
          else
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

void laganina_e_dejskras_a_mpi::TestMPITaskSequential::get_children_with_weights(
    int vertex, const std::vector<int>& row_ptr, const std::vector<int>& col_ind, const std::vector<int>& data,
    std::vector<int>& neighbor, std::vector<int>& weight) {
  // Get the beginning and end of edges for a given vertex
  int start = row_ptr[vertex];
  int end = row_ptr[vertex + 1];

  for (int i = start; i < end; ++i) {
    neighbor.push_back(col_ind[i]);  // Neighboring vertex
    weight.push_back(data[i]);       // Edge weight
  }
}

void laganina_e_dejskras_a_mpi::TestMPITaskSequential::dijkstra(int start_vertex, const std::vector<int>& row_ptr,
                                                                const std::vector<int>& col_ind,
                                                                const std::vector<int>& data, int v,
                                                                std::vector<int>& distances) {
  // Initialize distances
  distances.resize(v, std::numeric_limits<int>::max());
  distances[start_vertex] = 0;

  // Array to track visited vertices
  std::vector<bool> visited(v, false);

  // Priority queue for storing pairs (distance, vertex)
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>>
      priority_queue;
  priority_queue.emplace(0, start_vertex);  // Use start_vertex instead of 0

  while (!priority_queue.empty()) {
    int current_distance = priority_queue.top().first;
    int current_vertex = priority_queue.top().second;
    priority_queue.pop();

    // If the vertex has already been visited, skip it
    if (visited[current_vertex]) {
      continue;
    }

    // Mark the vertex as visited
    visited[current_vertex] = true;

    // Process all neighboring vertices
    int start = row_ptr[current_vertex];
    int end = row_ptr[current_vertex + 1];
    for (int i = start; i < end; ++i) {
      int neighbor_vertex = col_ind[i];
      int weight = data[i];
      int new_distance = current_distance + weight;

      // If a shorter distance is found, update it
      if (new_distance < distances[neighbor_vertex]) {
        distances[neighbor_vertex] = new_distance;
        priority_queue.emplace(new_distance, neighbor_vertex);
      }
    }
  }
}