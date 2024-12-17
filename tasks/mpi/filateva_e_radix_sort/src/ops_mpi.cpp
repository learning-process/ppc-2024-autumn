// Filateva Elizaveta Radix Sort
#include "mpi/filateva_e_radix_sort/include/ops_mpi.hpp"

#include <limits>
#include <string>
#include <boost/serialization/vector.hpp>
#include <list> 
#include <vector>

bool filateva_e_radix_sort_mpi::RadixSort::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    this->size = taskData->inputs_count[0];
    auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
    this->arr.assign(temp, temp + size);
    this->ans.resize(size);
  }
  return true;
}

bool filateva_e_radix_sort_mpi::RadixSort::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] <= 0 || taskData->inputs_count[0] != taskData->outputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool filateva_e_radix_sort_mpi::RadixSort::run() {
  internal_order_test();
  int kol = 20;
  int raz = 10;
  int delta;
  int ost;
  std::vector<int> local_ans;
  boost::mpi::broadcast(world, size, 0);
  if (world.rank() == 0) {
    delta = (world.size() == 1) ? 0 : arr.size() / (world.size() - 1);
    ost = (world.size() == 1) ? arr.size() : arr.size() % (world.size() - 1);
    local_ans.resize(size);
  }
  boost::mpi::broadcast(world, delta, 0);
  boost::mpi::broadcast(world, ost, 0);

  int local_size = (world.rank() == 0) ? ost : delta;
  std::vector<list<int>> radix_list(kol);
  std::vector<int> local_vec(local_size, 0);

  std::vector<int> distribution(world.size(), delta);
  distribution[0] = ost;
  std::vector<int> displacement(world.size(), 0);
  for (int i = 1; i < world.size(); i++) {
    displacement[i] = delta * (i - 1) + ost;
  }

  boost::mpi::scatterv(world, arr.data(), distribution, displacement, local_vec.data(), local_size, 0);
  for (int i = 0; i < local_size; i++) {
    radix_list[local_vec[i] % raz + 10].push_back(local_vec[i]);
  }
  while (radix_list[10].size() != local_size) {
    raz *= 10;
    std::vector<list<int>> temp(kol);
    for (int i = 0; i < kol; i++) {
      for (auto p : radix_list[i]) {
        temp[p % raz / (raz / 10) + 10].push_back(p);
      }
    }
    radix_list = temp;
  }
  int i = 0;
  for (auto a : radix_list[10]) {
    local_vec[i] = a;
    i++;
  }

  boost::mpi::gatherv(world, local_vec.data(), local_size, local_ans.data(), distribution, displacement, 0);

  if (world.rank() == 0) {
    std::vector<int> smesh(world.size(), 0);
    for (int j = 0; j < size; j++) {
      int min = -1;
      if (smesh[0] < ost) {
        min = 0;
      }
      for (int k = 1; k < world.size(); k++) {
        if (smesh[k] >= delta) {
          continue;
        }
        if (min == -1 || local_ans[displacement[k] + smesh[k]] < local_ans[displacement[min] + smesh[min]]) {
          min = k;
        }
      }
      ans[j] = local_ans[displacement[min] + smesh[min]];
      smesh[min]++;
    }
  }

  return true;
}

bool filateva_e_radix_sort_mpi::RadixSort::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(ans.begin(), ans.end(), output_data);
  }
  return true;
}
