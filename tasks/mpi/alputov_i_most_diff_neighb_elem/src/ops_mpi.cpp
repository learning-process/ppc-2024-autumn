// Copyright 2024 Alputov Ivan
#include "mpi/alputov_i_most_diff_neighb_elem/include/ops_mpi.hpp"

#include <algorithm>
#include <iostream>

std::vector<int> alputov_i_most_diff_neighb_elem_mpi::RandomVector(int sz) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-1000, 1000);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; ++i) {
    vec[i] = distrib(gen);
  }
  return vec;
}

int alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask::getElementsPerProcess() const {
  return taskData->inputs_count[0] / world.size();
}

int alputov_i_most_diff_neighb_elem_mpi::Max_Neighbour_Seq_Pos(const std::vector<int>& data) {
  if (data.size() < 2) {
    return 0;
  }
  int maxDifference = std::abs(data[0] - data[1]);
  int maxIndex = 0;
  for (size_t i = 1; i < data.size() - 1; ++i) {
    int difference = std::abs(data[i] - data[i + 1]);
    if (difference > maxDifference) {
      maxDifference = difference;
      maxIndex = i;
    }
  }
  return maxIndex;
}

bool alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 2;
}

bool alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask::pre_processing() {
  internal_order_test();
  int size = taskData->inputs_count[0];
  inputData = std::vector<int>(size);
  memcpy(inputData.data(), taskData->inputs[0], sizeof(int) * size);
  return true;
}

bool alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask::run() {
  internal_order_test();
  int index = Max_Neighbour_Seq_Pos(inputData);
  if (index == -1) {
    result[0] = 0;
    result[1] = 0;
    return false;
  } else {
    result[0] = inputData[index];
    result[1] = inputData[index + 1];
    return true;
  }
}
bool alputov_i_most_diff_neighb_elem_mpi::MPISequentialTask::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = result[0];
  reinterpret_cast<int*>(taskData->outputs[0])[1] = result[1];
  return true;
}

 void updateMaxDifferencePair(const int* currentPair, int* maxDiffPair, int* arrayLength, MPI_Datatype* dataType) {
  if (currentPair[2] > maxDiffPair[2]) {
    maxDiffPair[0] = currentPair[0];
    maxDiffPair[1] = currentPair[1];
    maxDiffPair[2] = currentPair[2];
  }
}
/* void updateMaxDifferencePair(const int* currentPair, int* maxDiffPair, int* arrayLength, MPI_Datatype* dataType) {
  if (currentPair[2] > maxDiffPair[2] ||
      (currentPair[2] == maxDiffPair[2] &&
       std::abs(currentPair[1] - currentPair[0]) > std::abs(maxDiffPair[1] - maxDiffPair[0]))) {
    memcpy(maxDiffPair, currentPair, 3 * sizeof(int));  // Копируем всю структуру
  }
}*/

int* findMaxDifference(const std::vector<int>& vec) {
  if (vec.size() < 2) {
    int* result = new int[3]{1, 1, 0};
    return result;
  }
  int* max_dif = new int[3];
  max_dif[0] = vec[0];
  max_dif[1] = vec[1];
  max_dif[2] = std::abs(vec[1] - vec[0]);
  for (size_t i = 1; i < vec.size() - 1; ++i) {
    int dif = std::abs(vec[i + 1] - vec[i]);
    if (dif > max_dif[2]) {
      max_dif[0] = vec[i];
      max_dif[1] = vec[i + 1];
      max_dif[2] = dif;
    }
  }
  return max_dif;
}

bool alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 2 && getElementsPerProcess() >= 2;
  }
  return true;
}

 /* bool alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask::pre_processing() {
  internal_order_test();
  int data_chunk = 0;
  int remainder = 0;
  if (world.rank() == 0) {
    data_chunk = taskData->inputs_count[0] / world.size();
    remainder = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, data_chunk, 0);
  broadcast(world, remainder, 0);

  int local_size = data_chunk;
  if (world.rank() == world.size() - 1) {
    local_size += remainder;
  }

  localData.resize(local_size);

  if (world.rank() == 0) {
    inputData = std::vector<int>(taskData->inputs_count[0]);
    memcpy(inputData.data(), taskData->inputs[0], sizeof(int) * taskData->inputs_count[0]);

    for (int proc = 1; proc < world.size(); ++proc) {
      int send_count = data_chunk;
      int offset = proc * data_chunk;
      if (proc == world.size() - 1) {
        send_count += remainder;
      }
      world.send(proc, 0, inputData.data() + offset, send_count);
    }
    localData = std::vector<int>(inputData.begin(), inputData.begin() + data_chunk);
  } else {
    world.recv(0, 0, localData.data(), local_size);
  }

  result[0] = 0;
  result[1] = 0;
  return true;
}*/

bool alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask::pre_processing() {
  internal_order_test();
  int data_chunk = 0;
  int remainder = 0;
  if (world.rank() == 0) {
    data_chunk = taskData->inputs_count[0] / world.size();
    remainder = taskData->inputs_count[0] % world.size();
  }
  broadcast(world, data_chunk, 0);
  broadcast(world, remainder, 0);

  int local_size = data_chunk;
  if (world.rank() < remainder) {
    local_size++;
  }

  localData.resize(local_size);

  if (world.rank() == 0) {
    inputData = std::vector<int>(taskData->inputs_count[0]);
    memcpy(inputData.data(), taskData->inputs[0], sizeof(int) * taskData->inputs_count[0]);

    int offset = local_size;  // Start offset for the next process
    for (int proc = 1; proc < world.size(); ++proc) {
      int send_count = data_chunk;
      if (proc < remainder) {
        send_count++;
      }
      world.send(proc, 0, inputData.data() + offset, send_count);
      offset += send_count;  // Update offset for the next process
    }
    std::copy(inputData.begin(), inputData.begin() + local_size, localData.begin());

  } else {
    world.recv(0, 0, localData.data(), local_size);
  }

  result[0] = 0;
  result[1] = 0;
  return true;
}

//bool alputov_i_most_diff_neighb_elem_mpi::mpiparalleltask::pre_processing() {
//  internal_order_test();
//  int data_chunk = 0;
//  int remainder = 0;
//
//  if (world.rank() == 0) {
//    data_chunk = taskdata->inputs_count[0] / world.size();
//    remainder = taskdata->inputs_count[0] % world.size();
//  }
//
//  broadcast(world, data_chunk, 0);
//  broadcast(world, remainder, 0);
//
//  int local_size = data_chunk + (world.rank() < remainder ? 1 : 0);
//  localdata.resize(local_size);
//
//  if (world.rank() == 0) {
//    inputdata = std::vector<int>(taskdata->inputs_count[0]);
//    memcpy(inputdata.data(), taskdata->inputs[0], sizeof(int) * taskdata->inputs_count[0]);
//
//    int offset = 0;
//    for (int proc = 1; proc < world.size(); ++proc) {
//      int send_count = data_chunk + (proc < remainder ? 1 : 0);
//      world.send(proc, 0, inputdata.data() + offset, send_count);
//      offset += send_count;
//    }
//    localdata = std::vector<int>(inputdata.begin(), inputdata.begin() + local_size);  //  исправлено
//  } else {
//    world.recv(0, 0, localdata.data(), local_size);
//  }
//
//  result[0] = 0;
//  result[1] = 0;
//  return true;
//}
//
//bool alputov_i_most_diff_neighb_elem_mpi::mpiparalleltask::run() {
//  internal_order_test();
//
//  int localmaxdata[3] = {0, 0, 0};  // новое имя локальной переменной
//
//  if (localdata.size() >= 2) {
//    int* localresult = findmaxdifference(localdata);
//    localmaxdata[0] = localresult[0];
//    localmaxdata[1] = localresult[1];
//    localmaxdata[2] = localresult[2];
//    delete[] localresult;
//
//    if (world.size() > 1) {
//      int lastelement = localdata.back();
//      int firstelement = localdata.front();
//      int nextfirstelement = 0;
//      int prevlastelement = 0;
//
//      int prevrank = (world.rank() > 0) ? world.rank() - 1 : world.size() - 1;
//      int nextrank = (world.rank() < world.size() - 1) ? world.rank() + 1 : 0;
//
//      boost::mpi::request reqs[4];
//      reqs[0] = world.isend(nextrank, 1, lastelement);
//      reqs[1] = world.irecv(prevrank, 0, prevlastelement);
//      reqs[2] = world.isend(prevrank, 0, firstelement);
//      reqs[3] = world.irecv(nextrank, 1, nextfirstelement);
//
//      boost::mpi::wait_all(reqs, reqs + 4);
//
//      int diff = std::abs(nextfirstelement - lastelement);
//      if (diff > localmaxdata[2]) {
//        localmaxdata[0] = lastelement;
//        localmaxdata[1] = nextfirstelement;
//        localmaxdata[2] = diff;
//      }
//
//      diff = std::abs(firstelement - prevlastelement);
//      if (diff > localmaxdata[2]) {
//        localmaxdata[0] = prevlastelement;
//        localmaxdata[1] = firstelement;
//        localmaxdata[2] = diff;
//      }
//    }
//
//    int globalmaxdiff[3] = {localmaxdata[0], localmaxdata[1], localmaxdata[2]};
//    mpi_op customoperation;
//    mpi_op_create(reinterpret_cast<mpi_user_function*>(&updatemaxdifferencepair), 1, &customoperation);
//    mpi_reduce(localmaxdata, globalmaxdiff, 3, mpi_int, customoperation, 0, mpi_comm_world);
//    mpi_op_free(&customoperation);
//
//    if (world.rank() == 0) {
//      result[0] = globalmaxdiff[0];
//      result[1] = globalmaxdiff[1];
//    }
//  } else {
//    if (world.rank() == 0) {
//      result[0] = 0;
//      result[1] = 0;
//    }
//  }
//  return true;
//}

 bool alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask::run() {
  internal_order_test();
  int* localResult = findMaxDifference(localData);
  if (localResult == nullptr) {
    return false;
  }
  localMaxDiff[0] = localResult[0];
  localMaxDiff[1] = localResult[1];
  localMaxDiff[2] = localResult[2];
  delete[] localResult;

  int lastElement = localData.back();
  int firstElement = localData.front();
  int nextFirstElement = 0;
  int prevLastElement = 0;
  if (world.rank() < world.size() - 1) {
    world.send(world.rank() + 1, 0, lastElement);
    world.recv(world.rank() + 1, 0, nextFirstElement);
  }
  if (world.rank() > 0) {
    world.send(world.rank() - 1, 0, firstElement);
    world.recv(world.rank() - 1, 0, prevLastElement);
  }
  if (world.rank() > 0) {
    int diff = std::abs(firstElement - prevLastElement);
    if (diff > localMaxDiff[2]) {
      localMaxDiff[0] = prevLastElement;
      localMaxDiff[1] = firstElement;
      localMaxDiff[2] = diff;
    }
  }
  if (world.rank() < world.size() - 1) {
    int diff = std::abs(nextFirstElement - lastElement);
    if (diff > localMaxDiff[2]) {
      localMaxDiff[0] = lastElement;
      localMaxDiff[1] = nextFirstElement;
      localMaxDiff[2] = diff;
    }
  }
  int globalDataArr[3] = {0, 0, 0};
  MPI_Op customOperation;
  MPI_Op_create(reinterpret_cast<MPI_User_function*>(&updateMaxDifferencePair), 1, &customOperation);
  MPI_Reduce(localMaxDiff, globalDataArr, 3, MPI_INT, customOperation, 0, MPI_COMM_WORLD);

  if (world.rank() == 0) {
    result[0] = globalDataArr[0];
    result[1] = globalDataArr[1];
  }
  MPI_Op_free(&customOperation);
  return true;
}

bool alputov_i_most_diff_neighb_elem_mpi::MPIParallelTask::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = result[0];
    reinterpret_cast<int*>(taskData->outputs[0])[1] = result[1];
  }
  return true;
}
