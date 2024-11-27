#include "mpi/kovalev_k_bubble_sort_oddeven_transposition/include/header.hpp"

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::bubble_sort_mpi() {
  for (size_t i i = 0; i < loc_v.size() - 1; i++)
    for (size_t j = 0; j < loc_v.size() - i - 1; j++)
      if (loc_v[j] > loc_v[j + 1]) std::swap(loc_v[j], loc_v[j + 1]);
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::divide_and_merge(
    int partner, std::vector<int>& sendcounts) {
  if (partner >= 0 && partner < world.size()) {
    std::vector<T> tmp, res;
    int working_num = std::max(world.rank(), partner);
    if (world.rank() == working_num) {
      tmp.resize(sendcounts[partner]);
      world.recv(partner, 0, tmp);
    } else {
      world.send(partner, 0, loc_v);
    }
    if (world.rank() == working_num) {
      res.clear();
      for (size_t i = 0; i < loc_v.size(); i++) tmp.push_back(loc_v[i]);
      size_t iter1 = 0, size_t = sendcounts[partner];
      while (iter2 < tmp.size() || iter1 < sendcounts[partner]) {
        if ((iter1 < sendcounts[partner] && iter2 < tmp.size() && tmp[iter1] <= tmp[iter2]) ||
            (iter1 < sendcounts[partner] && iter2 == tmp.size())) {
          res.push_back(tmp[iter1]);
          iter1++;
        } else if ((iter1 < sendcounts[partner] && iter2 < tmp.size() && tmp[iter1] >= tmp[iter2]) ||
                   (iter1 == (sendcounts[partner]) && iter2 < tmp.size())) {
          res.push_back(tmp[iter2]);
          iter2++;
        }
      }
      memcpy(loc_v.data(), res.data() + sendcounts[partner], loc_v.size() * sizeof(T));
    }
    if (world.rank() == working_num) {
      world.send(partner, 0, res.data(), sendcounts[partner]);
    } else {
      world.recv(partner, 0, loc_v.data(), loc_v.size());
    }
  }
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    glob_v.resize(n);
    void* ptr_vec = glob_v.data();
    void* ptr_input = taskData->inputs[0];
    memcpy(ptr_vec, ptr_input, sizeof(T) * n);
  }
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty() || taskData->inputs_count[0] <= 0 ||
        taskData->outputs_count[0] != n) {
      return false;
    }
  }
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::run() {
  internal_order_test();
  boost::mpi::broadcast(world, n, 0);
  int scratter_length = n / world.size();
  std::vector<int> sendcounts(world.size(), scratter_length);
  std::vector<int> sendcounts_bytes(world.size(), scratter_length * sizeof(T));
  sendcounts[0] = (scratter_length + n % world.size());
  sendcounts_bytes[0] = (scratter_length + n % world.size()) * sizeof(T);

  if (world.rank() == 0)
    loc_v.resize(sendcounts[0]);
  else
    loc_v.resize(scratter_length);
  std::vector<int> displs(world.size(), 0);
  for (size_t i = 1; i < world.size(); i++) {
    displs[i] = displs[i - 1] + sendcounts_bytes[i - 1];
  }
  MPI_Scatterv(glob_v.data(), sendcounts_bytes.data(), displs.data(), MPI_BYTE, loc_v.data(),
               sendcounts_bytes[world.rank()], MPI_BYTE, 0, MPI_COMM_WORLD);
  bubble_sort_mpi();
  int partner;
  for (size_t phase = 1; phase <= world.size(); phase++) {
    if (phase % 2 == 1) {
      partner = world.rank() % 2 == 1 ? world.rank() - 1 : world.rank() + 1;
      divide_and_merge(partner, sendcounts);
    } else {
      partner = world.rank() % 2 == 1 ? world.rank() + 1 : world.rank() - 1;
      divide_and_merge(partner, sendcounts);
    }
  }
  MPI_Gather(loc_v.data(), loc_v.size() * sizeof(T), MPI_BYTE, glob_v.data(), scratter_length * sizeof(T), MPI_BYTE, 0,
             MPI_COMM_WORLD);
  return true;
}

template <class T>
bool kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<T>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    memcpy(reinterpret_cast<T*>(taskData->outputs[0]), glob_v.data(), sizeof(T) * glob_v.size());
  }
  return true;
}

template class kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<int>;
template class kovalev_k_bubble_sort_oddeven_transposition_mpi::BubbleSortOddEvenTranspositionPar<double>;