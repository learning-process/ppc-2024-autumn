// Filateva Elizaveta Radix Sort
#include "mpi/filateva_e_radix_sort/include/ops_mpi.hpp"

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
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool filateva_e_radix_sort_mpi::RadixSort::run() {
  internal_order_test();
  int kol = 10;
  int raz = 10;
  std::vector<int> local_ans;

  boost::mpi::broadcast(world, size, 0);
  if (world.rank() == 0) {
    local_ans.resize(size);
  }

  int delta = (world.size() == 1) ? 0 : size / (world.size() - 1);
  int ost = (world.size() == 1) ? size : size % (world.size() - 1);
  int local_size = (world.rank() == 0) ? ost : delta;

  std::vector<std::list<int>> radix_list(kol);
  std::vector<std::list<int>> negativ_radix_list(kol);
  std::vector<int> local_vec(local_size, 0);

  std::vector<int> distribution(world.size(), delta);
  distribution[0] = ost;
  std::vector<int> displacement(world.size(), 0);
  for (int i = 1; i < world.size(); i++) {
    displacement[i] = delta * (i - 1) + ost;
  }

  boost::mpi::scatterv(world, arr.data(), distribution, displacement, local_vec.data(), local_size, 0);

  int max_r = -1;

  for (long unsigned int i = 0; i < local_vec.size(); i++) {
    max_r = std::max(max_r, std::abs(local_vec[i]));
    if (local_vec[i] >= 0) {
      radix_list[local_vec[i] % raz].push_back(local_vec[i]);
    } else {
      negativ_radix_list[std::abs(local_vec[i]) % raz].push_back(std::abs(local_vec[i]));
    }
  }
  while (max_r / (raz / 10) > 0) {
    raz *= 10;
    std::vector<std::list<int>> temp(kol);
    std::vector<std::list<int>> negativ_temp(kol);
    for (int i = 0; i < kol; i++) {
      for (auto p : radix_list[i]) {
        temp[p % raz / (raz / 10)].push_back(p);
      }
      for (auto p : negativ_radix_list[i]) {
        negativ_temp[p % raz / (raz / 10)].push_back(p);
      }
    }
    radix_list = temp;
    negativ_radix_list = negativ_temp;
  }

  auto rit = negativ_radix_list[0].rbegin();
  int i = 0;
  for (; rit != negativ_radix_list[0].rend(); rit++, i++) {
    local_vec[i] = -(*rit);
  }
  auto it = radix_list[0].begin();
  for (; it != radix_list[0].end(); it++, i++) {
    local_vec[i] = *it;
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
