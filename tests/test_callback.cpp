/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/random.h>
#include <memory>
#include <iostream>
#include <fstream>

TEST(TestCallback, timeout) {
    int n = 2000000;
    int k = 50000;
    int d = 1024;
//    int niter = 1; //1000000000;
    int seed = 42;

    std::vector<float> vecs(n * d);
    faiss::float_rand(vecs.data(), vecs.size(), seed);

//    auto index(new faiss::IndexFlat(d));

//    faiss::ClusteringParameters cp;
//    cp.niter = niter;
//    cp.verbose = true; //false;

//    faiss::Clustering kmeans(d, k, cp);

//    faiss::TimeoutCallback::reset(0.5);
//    EXPECT_THROW(kmeans.train(n, vecs.data(), *index), faiss::FaissException);

//    kmeans.train(n, vecs.data(), *index);

	std::unique_ptr<float[]> centrios(new float[d*k]);
	faiss::kmeans_clustering(d, n, k, vecs.data(), centrios.get());

#if 1
	std::ofstream outFile("/tmp/output_l2.txt");
	outFile << "The value of floatPtr is: \n";
	for (int i=0; i< d*k; i++)
	{
		outFile << centrios.get()[i] << "  ";
		if ((i+1)%10 == 0)
			outFile << "\n";
	}
	outFile << "\n";
	outFile.close();
#endif

//    delete index;
}
