// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/vladimirova_j_gather/include/ops_mpi.hpp"
#include "mpi/vladimirova_j_gather/include/ops_mpi_not_my_gather.hpp"
using namespace vladimirova_j_gather_mpi;
using namespace vladimirova_j_not_my_gather_mpi;
/*
TEST(Parallel_Operations_MPI, vladimirova_j_gather_1_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector =
  {2,2,-1,2,2,2,2,2,-1,2,2,2,-1,2,2,2,-1,-1,2,2,2,1,2,2,2,1,2,2,2,2,2,1,2,2,2,2,1,2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = { -1, -1, 2, 2, 1, 2 };
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
  }

  vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);


  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  ASSERT_EQ((int)taskDataPar->outputs_count[0], 6);
  std::cout << "!!!!!!!!!!!!!!!" << "\n";
  for (auto v: ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

 ASSERT_EQ(ans_buf_vec, ans_vec);

}

TEST(Parallel_Operations_MPI, vladimirova_j_gather_forward_backward_test) {
    boost::mpi::communicator world;
    std::vector<int> global_vector =
    {-2,2,-2,2,-2,2,-2,2,-2,2,-2,2,2 };
    //{0,1,2,3,4,5,6,7,8,9};
    std::vector<int32_t> ans_vec = { 2};
    std::vector<int32_t> ans_buf_vec(ans_vec.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
        taskDataPar->inputs_count.emplace_back(global_vector.size());
        taskDataPar->outputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    }

    vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
    std::cout << "!!!!!!!!!!!!!!!" << "\n";
    for (auto v : ans_buf_vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    ASSERT_EQ(ans_buf_vec, ans_vec);

}

TEST(Parallel_Operations_MPI, vladimirova_j_gather_right_left_test) {
    boost::mpi::communicator world;
    std::vector<int> global_vector =
    { -1,1,  -1,1, -1,1, -1 ,1 ,2 };
    //{0,1,2,3,4,5,6,7,8,9};
    std::vector<int32_t> ans_vec = {2};
    std::vector<int32_t> ans_buf_vec(ans_vec.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
        taskDataPar->inputs_count.emplace_back(global_vector.size());
        taskDataPar->outputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    }

    vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
    std::cout << "!!!!!!!!!!!!!!!" << "\n";
    for (auto v : ans_buf_vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    ASSERT_EQ(ans_buf_vec, ans_vec);

}
TEST(Parallel_Operations_MPI, vladimirova_j_gather_more_dead_ends_test) {
    boost::mpi::communicator world;
    std::vector<int> global_vector =
    {1,2,2,1,    2,1,-1,-1,-1,2,1,-2,   2,2,1,2,1,  2,-2  ,1,2,    1,-2,-1,-1,2,1,  2,1,     1,2,-1,-1,-2,1, 2 };
    //1 2 2    1 -1 -1 1    -2   2 2 1 2 1 2 -2 1 2 1 -2 -1 -1 2 1 1 -1 -2 1 2
    std::vector<int32_t> ans_vec = { 1, 2, 2,      2, 1, 2, 1, 1, 2, 1, -2, - 1, -1, 2, 1, -2, 1, 2 };
    std::vector<int32_t> ans_buf_vec(ans_vec.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
        taskDataPar->inputs_count.emplace_back(global_vector.size());
        taskDataPar->outputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    }

    vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    testMpiTaskParallel.validation();
    //ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
    std::cout << "!!!!!!!!!!!!!!!" << "\n";
    for (auto v : ans_buf_vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    ASSERT_EQ(ans_buf_vec, ans_vec);

}
TEST(Parallel_Operations_MPI, vladimirova_j_random_test) {
    boost::mpi::communicator world;
    std::vector<int> some_dead_end;
    std::vector<int> tmp;
    std::vector<int> global_vector;
    //{0,1,2,3,4,5,6,7,8,9};
    std::vector<int32_t> ans_vec = { -1, -1, 2, 2, 1, 2 };


    int noDEnd = 0;
    for (int j = 0; j < 10; j++) {
        some_dead_end = vladimirova_j_gather_mpi::getRandomVector(5);
        tmp = vladimirova_j_gather_mpi::getRandomVector(15);
        noDEnd += 15;
        global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
        global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
        global_vector.push_back(-1); global_vector.push_back(-1); noDEnd += 2;
        for (int i : some_dead_end) { if ((i != 2) && (i != -2)) i *= -1; }
        for (int i = some_dead_end.size() - 1; i >= 0; i--) global_vector.push_back(some_dead_end[i]);
    }

    std::vector<int32_t> ans_buf_vec(noDEnd);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
        taskDataPar->inputs_count.emplace_back(global_vector.size());
        taskDataPar->outputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    }

    vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    std::cout << "!!!!!!!!!!!!!!!" << "\n";
    for (auto v : ans_buf_vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    ASSERT_EQ((int)taskDataPar->outputs_count[0]<=noDEnd, true);
}
*/

/*
TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_1_test) {
    boost::mpi::communicator world;
    std::vector<int> global_vector =
    { 2,2,-1,2,2,2,2,2,-1,2,2,2,-1,2,2,2,-1,-1,2,2,2,1,2,2,2,1,2,2,2,2,2,1,2,2,2,2,1,2 };
    //{0,1,2,3,4,5,6,7,8,9};
    std::vector<int32_t> ans_vec = { -1, -1, 2, 2, 1, 2 };
    std::vector<int32_t> ans_buf_vec(ans_vec.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
        taskDataPar->inputs_count.emplace_back(global_vector.size());
        taskDataPar->outputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    }

    vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ((int)taskDataPar->outputs_count[0], 6);
    std::cout << "!!!!!!!!!!!!!!!" << "\n";
    for (auto v : ans_buf_vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    ASSERT_EQ(ans_buf_vec, ans_vec);

}

TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_1_test) {
    boost::mpi::communicator world;
    std::vector<int> global_vector =
    {0,1,2,3,4};
    //{0,1,2,3,4,5,6,7,8,9};
    std::vector<int32_t> ans_vec = { 0,0,0,0,0 };
    std::vector<int32_t> ans_buf_vec(ans_vec.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
        taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
        taskDataPar->inputs_count.emplace_back(global_vector.size());
        taskDataPar->outputs_count.emplace_back(1);
        taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    }

    vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    ASSERT_EQ((int)taskDataPar->outputs_count[0], 6);
    std::cout << "!!!!!!!!!!!!!!!" << "\n";
    for (auto v : ans_buf_vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    ASSERT_EQ(ans_buf_vec, ans_vec);

}

*/

TEST(Sequential_Operations_MPI, vladimirova_j_forward_backward_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {-2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataPar->inputs_count.emplace_back(global_vector.size());
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(ans_buf_vec, ans_vec);
}

TEST(Sequential_Operations_MPI, vladimirova_j_gather_right_left_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {-1, 1, -1, 1, -1, 1, -1, 1, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataPar->inputs_count.emplace_back(global_vector.size());
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(ans_buf_vec, ans_vec);
}
TEST(Sequential_Operations_MPI, vladimirova_j_gather_more_dead_ends_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {1,  2, 2, 1, 2,  1,  -1, -1, -1, 2, 1, -2, 2, 2,  1,  2,  1, 2,
                                    -2, 1, 2, 1, -2, -1, -1, 2,  1,  2, 1, 1,  2, -1, -1, -2, 1, 2};
  // 1 2 2    1 -1 -1 1    -2   2 2 1 2 1 2 -2 1 2 1 -2 -1 -1 2 1 1 -1 -2 1 2
  std::vector<int32_t> ans_vec = {1, 2, 2, 2, 1, 2, 1, 1, 2, 1, -2, -1, -1, 2, 1, -2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataPar->inputs_count.emplace_back(global_vector.size());
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  testMpiTaskSequential.validation();
  // ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  ASSERT_EQ((int)taskDataPar->outputs_count[0], ans_vec.size());
  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(ans_buf_vec, ans_vec);
}
TEST(Sequential_Operations_MPI, vladimirova_j_random_test) {
  boost::mpi::communicator world;
  std::vector<int> some_dead_end;
  std::vector<int> tmp;
  std::vector<int> global_vector;
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};

  int noDEnd = 0;
  for (int j = 0; j < 10; j++) {
    some_dead_end = vladimirova_j_gather_mpi::getRandomVector(5);
    tmp = vladimirova_j_gather_mpi::getRandomVector(15);
    noDEnd += 15;
    global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
    global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
    global_vector.push_back(-1);
    global_vector.push_back(-1);
    noDEnd += 2;
    for (int i : some_dead_end) {
      if ((i != 2) && (i != -2)) i *= -1;
    }
    for (int i = some_dead_end.size() - 1; i >= 0; i--) global_vector.push_back(some_dead_end[i]);
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataPar->inputs_count.emplace_back(global_vector.size());
  taskDataPar->outputs_count.emplace_back(1);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ((int)taskDataPar->outputs_count[0] <= noDEnd, true);
}

TEST(Sequential_Operations_MPI, vladimirova_j_not_gather_1_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {2, 2, -1, 2, 2, 2, 2, 2, -1, 2, 2, 2, -1, 2, 2, 2, -1, -1, 2,
                                    2, 2, 1,  2, 2, 2, 1, 2, 2,  2, 2, 2, 1,  2, 2, 2, 2,  1,  2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ((int)taskDataSeq->outputs_count[0], ans_vec.size());
  std::cout << "!!!!!!!!!!!!!!!"
            << "\n";
  for (auto v : ans_buf_vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;

  ASSERT_EQ(ans_buf_vec, ans_vec);
}
