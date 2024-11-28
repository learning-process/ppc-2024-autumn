#include "mpi/zaitsev_a_scatter/include/ops_mpi.hpp"

int zaitsev_a_scatter::scatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                               MPI_Datatype recvtype, int root, MPI_Comm comm) {
  if (sendcount != recvcount) return MPI_ERR_COUNT;
  if (sendtype != recvtype) return MPI_ERR_TYPE;

  int commsize;
  MPI_Comm_size(comm, &commsize);

  int rank;
  MPI_Comm_rank(comm, &rank);

  int sendsize;
  MPI_Type_size(sendtype, &sendsize);

  int recvsize;
  MPI_Type_size(recvtype, &recvsize);

  if (rank == root) {
    char* p = (char*)sendbuf;
    const auto do_send = [&](int begin, int end) {
      for (int i = begin; i < end; i++) {
        MPI_Send(p, sendcount, sendtype, i, 0, comm);
        p += sendcount * recvsize;
      }
    };
    do_send(0, root);
    do_send(root + 1, commsize);
    memcpy(recvbuf, sendbuf, recvcount * recvsize);
  } else {
    MPI_Status status;
    MPI_Recv(recvbuf, recvcount, recvtype, root, 0, comm, &status);
  }

  return MPI_SUCCESS;
}