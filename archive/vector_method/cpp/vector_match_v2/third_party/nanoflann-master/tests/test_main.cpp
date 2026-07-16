/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2011-2026 Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>  // for abs()
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <nanoflann.hpp>
#include <random>
#include <set>
#include <vector>

#include "../examples/utils.h"

using namespace std;
using namespace nanoflann;

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

template <typename num_t>
void L2_vs_L2_simple_test(const size_t N, const size_t num_results)
{
    PointCloud<num_t> cloud;

    // Generate points:
    generateRandomPointCloud(cloud, N);

    num_t query_pt[3] = {0.5, 0.5, 0.5};

    // construct a kd-tree index:
    using my_kd_tree_simple_t = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 3 /* dim */
        >;

    using my_kd_tree_t = KDTreeSingleIndexAdaptor<
        L2_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 3 /* dim */
        >;

    my_kd_tree_simple_t index1(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    my_kd_tree_t index2(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    // do a knn search
    std::vector<size_t>            ret_index(num_results);
    std::vector<num_t>             out_dist_sqr(num_results);
    nanoflann::KNNResultSet<num_t> resultSet(num_results);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);
    index1.findNeighbors(resultSet, &query_pt[0]);

    std::vector<size_t> ret_index1    = ret_index;
    std::vector<num_t>  out_dist_sqr1 = out_dist_sqr;

    resultSet.init(&ret_index[0], &out_dist_sqr[0]);

    index2.findNeighbors(resultSet, &query_pt[0]);

    if (N >= num_results)
    {
        EXPECT_EQ(resultSet.size(), num_results);
    }
    else
    {
        EXPECT_EQ(resultSet.size(), N);
    }

    for (size_t i = 0; i < resultSet.size(); i++)
    {
        EXPECT_EQ(ret_index1[i], ret_index[i]);
        EXPECT_NEAR(out_dist_sqr1[i], out_dist_sqr[i], 1e-3);
    }
    // Ensure results are sorted:
    num_t lastDist = -1;
    for (size_t i = 0; i < resultSet.size(); i++)
    {
        const num_t newDist = out_dist_sqr[i];
        EXPECT_GE(newDist, lastDist);
        lastDist = newDist;
    }

    // Test "RadiusResultSet" too:
    const num_t maxRadiusSqrSearch = 10.0 * 10.0;

    std::vector<nanoflann::ResultItem<
        typename my_kd_tree_simple_t::IndexType, typename my_kd_tree_simple_t::DistanceType>>
        radiusIdxs;

    nanoflann::RadiusResultSet<num_t, typename my_kd_tree_simple_t::IndexType> radiusResults(
        maxRadiusSqrSearch, radiusIdxs);
    radiusResults.init();
    nanoflann::SearchParameters searchParams;
    searchParams.sorted = true;
    index1.findNeighbors(radiusResults, &query_pt[0], searchParams);

    // Ensure results are sorted:
    lastDist = -1;
    for (const auto& r : radiusIdxs)
    {
        const num_t newDist = r.second;
        EXPECT_GE(newDist, lastDist);
        lastDist = newDist;
    }
}

using namespace nanoflann;
#include "../examples/KDTreeVectorOfVectorsAdaptor.h"

template <typename NUM>
void generateRandomPointCloud(
    std::vector<std::vector<NUM>>& samples, const size_t N, const size_t dim, const NUM max_range)
{
    samples.resize(N);
    for (size_t i = 0; i < N; i++)
    {
        samples[i].resize(dim);
        for (size_t d = 0; d < dim; d++)
            samples[i][d] = max_range * NUM(rand() % 1000) / NUM(1000.0);
    }
}

template <typename NUM>
void L1_vs_bruteforce_test(const size_t nSamples, const size_t DIM, const size_t numToSearch)
{
    std::vector<std::vector<NUM>> samples;

    const NUM max_range = NUM(20.0);

    // Generate points:
    generateRandomPointCloud(samples, nSamples, DIM, max_range);

    // Query point:
    std::vector<NUM> query_pt(DIM);
    for (size_t d = 0; d < DIM; d++) query_pt[d] = max_range * NUM(rand() % 1000) / NUM(1000.0);

    // construct a kd-tree index:
    // Dimensionality set at run-time
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor<
        std::vector<std::vector<NUM>>, NUM, -1, nanoflann::metric_L1>
        my_kd_tree_t;

    my_kd_tree_t mat_index(DIM /*dim*/, samples, 10 /* max leaf */);

    // do a knn search
    const size_t        num_results = numToSearch;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<NUM>    out_dists_sqr(num_results);

    nanoflann::KNNResultSet<NUM> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    mat_index.index->findNeighbors(resultSet, &query_pt[0]);

    const auto nFound = resultSet.size();

    EXPECT_TRUE(nFound > 0);
    if (resultSet.full())
    {
        EXPECT_EQ(resultSet.worstDist(), out_dists_sqr.at(nFound - 1));
    }

    // Brute force neighbors:
    std::multimap<NUM /*dist*/, size_t /*idx*/> bf_nn;
    {
        for (size_t i = 0; i < nSamples; i++)
        {
            double dist = 0.0;
            for (size_t d = 0; d < DIM; d++) dist += std::abs(query_pt[d] - samples[i][d]);
            bf_nn.emplace(dist, i);
        }
    }

    // Keep bruteforce solutions indexed by idx instead of distances,
    // to handle correctly almost or exactly coindicing distances for >=2 NN:
    std::map<size_t, NUM> bf_idx2dist;
    for (const auto& kv : bf_nn) bf_idx2dist[kv.second] = kv.first;

    // Compare:
    if (!bf_nn.empty())
    {
        auto it = bf_nn.begin();
        for (size_t i = 0; i < nFound; ++i, ++it)
        {
            // Distances must be in exact order:
            EXPECT_NEAR(it->first, out_dists_sqr[i], 1e-3);

            // indices may be not in the (rare) case of a tie:
            EXPECT_NEAR(bf_idx2dist.at(ret_indexes[i]), out_dists_sqr[i], 1e-3)
                << "For: numToSearch=" << numToSearch << " out_dists_sqr[i]=" << out_dists_sqr[i]
                << "\n";
        }
    }
}

template <typename NUM>
void rknn_L1_vs_bruteforce_test(
    const size_t nSamples, const size_t DIM, const size_t numToSearch, const NUM maxRadiusSqr)
{
    std::vector<std::vector<NUM>> samples;

    const NUM max_range = NUM(20.0);

    // Generate points:
    generateRandomPointCloud(samples, nSamples, DIM, max_range);

    // Query point:
    std::vector<NUM> query_pt(DIM);
    for (size_t d = 0; d < DIM; d++) query_pt[d] = max_range * NUM(rand() % 1000) / NUM(1000.0);

    // construct a kd-tree index:
    // Dimensionality set at run-time
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor<
        std::vector<std::vector<NUM>>, NUM, -1, nanoflann::metric_L1>
        my_kd_tree_t;

    my_kd_tree_t mat_index(DIM /*dim*/, samples, 10 /* max leaf */);

    // do a knn search
    const size_t        num_results = numToSearch;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<NUM>    out_dists_sqr(num_results);

    nanoflann::RKNNResultSet<NUM> resultSet(num_results, maxRadiusSqr);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    mat_index.index->findNeighbors(resultSet, &query_pt[0]);

    const auto nFound = resultSet.size();

    if (resultSet.full())
    {
        EXPECT_EQ(resultSet.worstDist(), out_dists_sqr.at(nFound - 1));
    }

    // Brute force neighbors:
    std::multimap<NUM /*dist*/, size_t /*idx*/> bf_nn;
    {
        for (size_t i = 0; i < nSamples; i++)
        {
            NUM dist = NUM(0.0);
            for (size_t d = 0; d < DIM; d++)
                dist += static_cast<NUM>(std::abs(query_pt[d] - samples[i][d]));

            if (dist < maxRadiusSqr) bf_nn.emplace(dist, i);
        }
    }

    // Keep bruteforce solutions indexed by idx instead of distances,
    // to handle correctly almost or exactly coindicing distances for >=2 NN:
    std::map<size_t, NUM> bf_idx2dist;
    for (const auto& kv : bf_nn) bf_idx2dist[kv.second] = kv.first;

    // Compare:
    if (!bf_nn.empty())
    {
        const size_t compareCount = std::min<size_t>(nFound, bf_nn.size());
        auto         it           = bf_nn.begin();
        for (size_t i = 0; i < compareCount; ++i, ++it)
        {
            // Distances must be in exact order:
            EXPECT_NEAR(it->first, out_dists_sqr[i], 1e-3)
                << "For: numToSearch=" << numToSearch << " out_dists_sqr[i]=" << out_dists_sqr[i]
                << "\n";

            if (bf_idx2dist.find(ret_indexes[i]) != bf_idx2dist.end())
            {
                EXPECT_NEAR(bf_idx2dist[ret_indexes[i]], out_dists_sqr[i], 1e-3)
                    << "For: numToSearch=" << numToSearch
                    << " out_dists_sqr[i]=" << out_dists_sqr[i] << "\n";
            }
        }
    }
}

template <typename NUM>
void L2_vs_bruteforce_test(const size_t nSamples, const size_t DIM, const size_t numToSearch)
{
    std::vector<std::vector<NUM>> samples;

    const NUM max_range = NUM(20.0);

    // Generate points:
    generateRandomPointCloud(samples, nSamples, DIM, max_range);

    // Query point:
    std::vector<NUM> query_pt(DIM);
    for (size_t d = 0; d < DIM; d++) query_pt[d] = max_range * NUM(rand() % 1000) / NUM(1000.0);

    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<NUM>>, NUM> my_kd_tree_t;

    my_kd_tree_t mat_index(DIM /*dim*/, samples, 10 /* max leaf */);

    // do a knn search
    const size_t        num_results = numToSearch;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<NUM>    out_dists_sqr(num_results);

    nanoflann::KNNResultSet<NUM> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    mat_index.index->findNeighbors(resultSet, &query_pt[0]);

    const auto nFound = resultSet.size();

    EXPECT_TRUE(nFound > 0);
    if (resultSet.full())
    {
        EXPECT_EQ(resultSet.worstDist(), out_dists_sqr.at(nFound - 1));
    }

    // Brute force neighbors:
    std::multimap<NUM /*dist*/, size_t /*idx*/> bf_nn;
    {
        for (size_t i = 0; i < nSamples; i++)
        {
            double dist = 0.0;
            for (size_t d = 0; d < DIM; d++)
                dist += (query_pt[d] - samples[i][d]) * (query_pt[d] - samples[i][d]);
            bf_nn.emplace(dist, i);
        }
    }

    // Keep bruteforce solutions indexed by idx instead of distances,
    // to handle correctly almost or exactly coindicing distances for >=2 NN:
    std::map<size_t, NUM> bf_idx2dist;
    for (const auto& kv : bf_nn) bf_idx2dist[kv.second] = kv.first;

    // Compare:
    if (!bf_nn.empty())
    {
        auto it = bf_nn.begin();
        for (size_t i = 0; i < nFound; ++i, ++it)
        {
            // Distances must be in exact order:
            EXPECT_NEAR(it->first, out_dists_sqr[i], 1e-3);

            // indices may be not in the (rare) case of a tie:
            EXPECT_NEAR(bf_idx2dist.at(ret_indexes[i]), out_dists_sqr[i], 1e-3)
                << "For: numToSearch=" << numToSearch << " out_dists_sqr[i]=" << out_dists_sqr[i]
                << "\n";
        }
    }
}

template <typename NUM>
void box_L2_vs_bruteforce_test(const size_t nSamples, const size_t DIM)
{
    std::vector<std::vector<NUM>> samples;

    const NUM max_range = NUM(20.0);

    // Generate points:
    generateRandomPointCloud(samples, nSamples, DIM, max_range);

    typedef KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<NUM>>, NUM> my_kd_tree_t;

    // Query box:
    typename my_kd_tree_t::index_t::BoundingBox query_box(DIM);
    for (size_t d = 0; d < DIM; d++)
    {
        query_box[d].low  = max_range * NUM(rand() % 1000) / NUM(1000.0);
        query_box[d].high = max_range * NUM(rand() % 1000) / NUM(1000.0);
        if (query_box[d].low > query_box[d].high) std::swap(query_box[d].low, query_box[d].high);
    }

    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    my_kd_tree_t mat_index(DIM /*dim*/, samples, 10 /* max leaf */);

    // do a knn search
    std::vector<size_t> ret_indexes(nSamples);
    std::vector<NUM>    out_dists_sqr(nSamples);

    nanoflann::KNNResultSet<NUM> resultSet(nSamples);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    const auto nFound = mat_index.index->findWithinBox(resultSet, query_box);

    // Brute force:
    std::set<size_t /*idx*/> bf_nn;
    for (size_t i = 0; i < nSamples; i++)
    {
        if (mat_index.index->contains(query_box, i)) bf_nn.insert(i);
    }

    // Compare:
    EXPECT_EQ(bf_nn.size(), nFound);

    for (size_t i = 0; i < nFound; ++i)
    {
        EXPECT_TRUE(bf_nn.find(ret_indexes[i]) != bf_nn.end());
    }
}

template <typename NUM>
void rknn_L2_vs_bruteforce_test(
    const size_t nSamples, const size_t DIM, const size_t numToSearch, const NUM maxRadiusSqr)
{
    std::vector<std::vector<NUM>> samples;

    const NUM max_range = NUM(20.0);

    // Generate points:
    generateRandomPointCloud(samples, nSamples, DIM, max_range);

    // Query point:
    std::vector<NUM> query_pt(DIM);
    for (size_t d = 0; d < DIM; d++) query_pt[d] = max_range * NUM(rand() % 1000) / NUM(1000.0);

    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<NUM>>, NUM> my_kd_tree_t;

    my_kd_tree_t mat_index(DIM /*dim*/, samples, 10 /* max leaf */);

    // do a knn search
    const size_t        num_results = numToSearch;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<NUM>    out_dists_sqr(num_results);

    nanoflann::RKNNResultSet<NUM> resultSet(num_results, maxRadiusSqr);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    mat_index.index->findNeighbors(resultSet, &query_pt[0]);

    const auto nFound = resultSet.size();

    if (resultSet.full())
    {
        EXPECT_EQ(resultSet.worstDist(), out_dists_sqr.at(nFound - 1));
    }

    // Brute force neighbors:
    std::multimap<NUM /*dist*/, size_t /*idx*/> bf_nn;
    {
        for (size_t i = 0; i < nSamples; i++)
        {
            double dist = 0.0;
            for (size_t d = 0; d < DIM; d++)
                dist += (query_pt[d] - samples[i][d]) * (query_pt[d] - samples[i][d]);

            if (dist <= maxRadiusSqr) bf_nn.emplace(dist, i);
        }
    }

    // Keep bruteforce solutions indexed by idx instead of distances,
    // to handle correctly almost or exactly coindicing distances for >=2 NN:
    std::map<size_t, NUM> bf_idx2dist;
    for (const auto& kv : bf_nn) bf_idx2dist[kv.second] = kv.first;

    // Compare:
    if (!bf_nn.empty())
    {
        auto it = bf_nn.begin();
        for (size_t i = 0; i < nFound; ++i, ++it)
        {
            // Distances must be in exact order:
            EXPECT_NEAR(it->first, out_dists_sqr[i], 1e-3)
                << "For: numToSearch=" << numToSearch << " out_dists_sqr[i]=" << out_dists_sqr[i]
                << "\n";

            // indices may be not in the (rare) case of a tie:
            EXPECT_NEAR(bf_idx2dist.at(ret_indexes[i]), out_dists_sqr[i], 1e-3)
                << "For: numToSearch=" << numToSearch << " out_dists_sqr[i]=" << out_dists_sqr[i]
                << "\n";
        }
    }
}

template <typename NUM>
void SO3_vs_bruteforce_test(const size_t nSamples)
{
    PointCloud_Quat<NUM> cloud;

    // Generate points:
    generateRandomPointCloud_Quat(cloud, nSamples);

    NUM query_pt[4] = {0.5, 0.5, 0.5, 0.5};

    // construct a kd-tree index:
    typedef KDTreeSingleIndexAdaptor<
        SO3_Adaptor<NUM, PointCloud_Quat<NUM>>, PointCloud_Quat<NUM>, 4 /* dim */
        >
        my_kd_tree_t;

    my_kd_tree_t index(4 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    // do a knn search
    const size_t        num_results = 1;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<NUM>    out_dists_sqr(num_results);

    nanoflann::KNNResultSet<NUM> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    index.findNeighbors(resultSet, &query_pt[0]);

    // Brute force:
    double min_dist_L2 = std::numeric_limits<double>::max();
    size_t min_idx     = std::numeric_limits<size_t>::max();
    {
        for (size_t i = 0; i < nSamples; i++)
        {
            double dist = 0.0;
            for (int d = 0; d < 4; d++)
                dist += (query_pt[d] - cloud.kdtree_get_pt(i, d)) *
                        (query_pt[d] - cloud.kdtree_get_pt(i, d));
            if (dist < min_dist_L2)
            {
                min_dist_L2 = dist;
                min_idx     = i;
            }
        }
        ASSERT_TRUE(min_idx != std::numeric_limits<size_t>::max());
    }

    // Compare:
    EXPECT_EQ(min_idx, ret_indexes[0]);
    EXPECT_NEAR(min_dist_L2, out_dists_sqr[0], 1e-3);
}

template <typename NUM>
void SO2_vs_bruteforce_test(const size_t nSamples)
{
    PointCloud_Orient<NUM> cloud;

    // Generate points:
    generateRandomPointCloud_Orient(cloud, nSamples);

    NUM query_pt[1] = {0.5};

    // construct a kd-tree index:
    typedef KDTreeSingleIndexAdaptor<
        SO2_Adaptor<NUM, PointCloud_Orient<NUM>>, PointCloud_Orient<NUM>, 1 /* dim */
        >
        my_kd_tree_t;

    my_kd_tree_t index(1 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    // do a knn search
    const size_t        num_results = 1;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<NUM>    out_dists_sqr(num_results);

    nanoflann::KNNResultSet<NUM> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    index.findNeighbors(resultSet, &query_pt[0]);

    // Brute force: find nearest neighbour by absolute angular distance,
    // matching SO2_Adaptor::accum_dist which returns |b - a| wrapped.
    double min_dist_SO2 = std::numeric_limits<double>::max();
    size_t min_idx      = std::numeric_limits<size_t>::max();
    {
        for (size_t i = 0; i < nSamples; i++)
        {
            double dist = cloud.kdtree_get_pt(i, 0) - query_pt[0];
            if (dist > nanoflann::pi_const<double>())
                dist -= 2 * nanoflann::pi_const<double>();
            else if (dist < -nanoflann::pi_const<double>())
                dist += 2 * nanoflann::pi_const<double>();
            if (dist < 0) dist = -dist;  // absolute angular distance
            if (dist < min_dist_SO2)
            {
                min_dist_SO2 = dist;
                min_idx      = i;
            }
        }
        ASSERT_TRUE(min_idx != std::numeric_limits<size_t>::max());
    }
    // Compare: the kd-tree must return *a* nearest neighbour, i.e. one whose
    // angular distance equals the brute-force minimum. We must not require the
    // index to match min_idx exactly: when two points are (nearly) equidistant
    // from the query, the brute-force loop keeps the lowest index while the
    // kd-tree may return any of the tied points, so an index comparison is
    // ill-defined under ties.
    SO2_Adaptor<NUM, PointCloud_Orient<NUM>> so2(cloud);
    const double                             ret_dist =
        so2.accum_dist(query_pt[0], cloud.kdtree_get_pt(ret_indexes[0], 0), 0);
    EXPECT_NEAR(min_dist_SO2, ret_dist, 1e-4);
    EXPECT_NEAR(min_dist_SO2, out_dists_sqr[0], 1e-3);
}

// First add nSamples/2 points, find the closest point
// Then add remaining points and find closest point
// Compare both with closest point using brute force approach
template <typename NUM>
void L2_dynamic_vs_bruteforce_test(const size_t nSamples)
{
    PointCloud<NUM> cloud;

    const NUM max_range = NUM(20.0);

    // construct a kd-tree index:
    typedef KDTreeSingleIndexDynamicAdaptor<
        L2_Simple_Adaptor<NUM, PointCloud<NUM>>, PointCloud<NUM>, 3 /* dim */
        >
        my_kd_tree_t;

    my_kd_tree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    // Generate points:
    generateRandomPointCloud(cloud, nSamples, max_range);

    NUM query_pt[3] = {0.5, 0.5, 0.5};

    // add points in chunks at a time
    size_t chunk_size = 100;
    size_t end        = 0;
    for (size_t i = 0; i < nSamples / 2; i = i + chunk_size)
    {
        end = min(size_t(i + chunk_size), nSamples / 2 - 1);
        index.addPoints(static_cast<uint32_t>(i), static_cast<uint32_t>(end));
    }

    {
        // do a knn search
        const size_t        num_results = 1;
        std::vector<size_t> ret_indexes(num_results);
        std::vector<NUM>    out_dists_sqr(num_results);

        nanoflann::KNNResultSet<NUM> resultSet(num_results);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        index.findNeighbors(resultSet, &query_pt[0]);

        // Brute force:
        double min_dist_L2 = std::numeric_limits<double>::max();
        size_t min_idx     = std::numeric_limits<size_t>::max();
        {
            for (size_t i = 0; i < nSamples / 2; i++)
            {
                double dist = 0.0;
                for (int d = 0; d < 3; d++)
                    dist += (query_pt[d] - cloud.kdtree_get_pt(i, d)) *
                            (query_pt[d] - cloud.kdtree_get_pt(i, d));
                if (dist < min_dist_L2)
                {
                    min_dist_L2 = dist;
                    min_idx     = i;
                }
            }
            ASSERT_TRUE(min_idx != std::numeric_limits<size_t>::max());
        }
        // Compare:
        EXPECT_EQ(min_idx, ret_indexes[0]);
        EXPECT_NEAR(min_dist_L2, out_dists_sqr[0], 1e-3);
    }
    for (size_t i = end + 1; i < nSamples; i = i + chunk_size)
    {
        end = min(size_t(i + chunk_size), nSamples - 1);
        index.addPoints(static_cast<uint32_t>(i), static_cast<uint32_t>(end));
    }

    {
        // do a knn search
        const size_t        num_results = 1;
        std::vector<size_t> ret_indexes(num_results);
        std::vector<NUM>    out_dists_sqr(num_results);

        nanoflann::KNNResultSet<NUM> resultSet(num_results);

        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        index.findNeighbors(resultSet, &query_pt[0]);

        // Brute force:
        double min_dist_L2 = std::numeric_limits<double>::max();
        size_t min_idx     = std::numeric_limits<size_t>::max();
        {
            for (size_t i = 0; i < nSamples; i++)
            {
                double dist = 0.0;
                for (int d = 0; d < 3; d++)
                    dist += (query_pt[d] - cloud.kdtree_get_pt(i, d)) *
                            (query_pt[d] - cloud.kdtree_get_pt(i, d));
                if (dist < min_dist_L2)
                {
                    min_dist_L2 = dist;
                    min_idx     = i;
                }
            }
            ASSERT_TRUE(min_idx != std::numeric_limits<size_t>::max());
        }
        // Compare:
        EXPECT_EQ(min_idx, ret_indexes[0]);
        EXPECT_NEAR(min_dist_L2, out_dists_sqr[0], 1e-3);
    }
}

template <typename NUM>
void L2_concurrent_build_vs_bruteforce_test(const size_t nSamples, const size_t DIM)
{
    std::vector<std::vector<NUM>> samples;

    const NUM max_range = NUM(20.0);

    // Generate points:
    generateRandomPointCloud(samples, nSamples, DIM, max_range);

    // Query point:
    std::vector<NUM> query_pt(DIM);
    for (size_t d = 0; d < DIM; d++) query_pt[d] = max_range * NUM(rand() % 1000) / NUM(1000.0);

    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<NUM>>, NUM> my_kd_tree_t;

    my_kd_tree_t mat_index(DIM /*dim*/, samples, 10 /* max leaf */, 0 /* concurrent build */);

    // do a knn search
    const size_t        num_results = 1;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<NUM>    out_dists_sqr(num_results);

    nanoflann::KNNResultSet<NUM> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    mat_index.index->findNeighbors(resultSet, &query_pt[0]);

    // Brute force:
    double min_dist_L2 = std::numeric_limits<double>::max();
    size_t min_idx     = std::numeric_limits<size_t>::max();
    {
        for (size_t i = 0; i < nSamples; i++)
        {
            double dist = 0.0;
            for (size_t d = 0; d < DIM; d++)
                dist += (query_pt[d] - samples[i][d]) * (query_pt[d] - samples[i][d]);
            if (dist < min_dist_L2)
            {
                min_dist_L2 = dist;
                min_idx     = i;
            }
        }
        ASSERT_TRUE(min_idx != std::numeric_limits<size_t>::max());
    }

    // Compare:
    EXPECT_EQ(min_idx, ret_indexes[0]);
    EXPECT_NEAR(min_dist_L2, out_dists_sqr[0], 1e-3);
}

template <typename NUM>
void L2_concurrent_build_vs_L2_test(const size_t nSamples, const size_t DIM)
{
    std::vector<std::vector<NUM>> samples;

    const NUM max_range = NUM(20.0);

    // Generate points:
    generateRandomPointCloud(samples, nSamples, DIM, max_range);

    // Query point:
    std::vector<NUM> query_pt(DIM);
    for (size_t d = 0; d < DIM; d++) query_pt[d] = max_range * NUM(rand() % 1000) / NUM(1000.0);

    // construct a kd-tree index:
    // Dimensionality set at run-time (default: L2)
    // ------------------------------------------------------------
    typedef KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<NUM>>, NUM> my_kd_tree_t;

    my_kd_tree_t mat_index_concurrent_build(
        DIM /*dim*/, samples, 10 /* max leaf */, 0 /* concurrent build */);
    my_kd_tree_t mat_index(DIM /*dim*/, samples, 10 /* max leaf */);

    // Compare:
    EXPECT_EQ(mat_index.index->vAcc_, mat_index_concurrent_build.index->vAcc_);
}

template <typename num_t>
void L2_dynamic_sorted_test(const size_t N, const size_t num_results)
{
    if (num_results == 0) return;

    PointCloud<num_t> cloud;
    generateRandomPointCloud(cloud, N);

    num_t query_pt[3] = {0.5, 0.5, 0.5};

    using DynamicKDTree = KDTreeSingleIndexDynamicAdaptor<
        L2_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 3 /* dim */
        >;

    DynamicKDTree dynamic_index(3, cloud);

    // Prepare result containers
    std::vector<size_t>                    dynamic_idx(num_results);
    std::vector<num_t>                     dynamic_dist(num_results);
    KNNResultSet<num_t>                    dynamic_knn_result(num_results);
    std::vector<ResultItem<size_t, num_t>> radius_results_vec;
    RadiusResultSet<num_t, size_t>         dynamic_radius_result(10.0 * 10.0, radius_results_vec);

    // Prepare search params to sort result
    const auto search_params = SearchParameters(0, true);

    dynamic_knn_result.init(&dynamic_idx[0], &dynamic_dist[0]);
    dynamic_radius_result.init();

    dynamic_index.findNeighbors(dynamic_knn_result, &query_pt[0], search_params);
    dynamic_index.findNeighbors(dynamic_radius_result, &query_pt[0], search_params);

    // Check size
    const size_t expected_size = std::min(N, num_results);
    ASSERT_EQ(dynamic_knn_result.size(), expected_size);

    // Ensure knn results are sorted
    num_t last_dist = -1;
    for (size_t i = 0; i < expected_size; i++)
    {
        EXPECT_GE(dynamic_dist[i], last_dist);
        last_dist = dynamic_dist[i];
    }

    // Ensure radius results are sorted
    num_t last = -1;
    for (const auto& r : radius_results_vec)
    {
        EXPECT_GE(r.second, last);
        last = r.second;
    }
}

TEST(kdtree, L1_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int knn = 1; knn < 20; knn += 3)
    {
        for (int i = 0; i < 500; i++)
        {
            L1_vs_bruteforce_test<float>(10, 2, knn);

            L1_vs_bruteforce_test<float>(100, 2, knn);
            L1_vs_bruteforce_test<float>(100, 3, knn);
            L1_vs_bruteforce_test<float>(100, 7, knn);

            L1_vs_bruteforce_test<double>(100, 2, knn);
            L1_vs_bruteforce_test<double>(100, 3, knn);
            L1_vs_bruteforce_test<double>(100, 7, knn);
        }
    }
}

TEST(kdtree, L1_vs_bruteforce_rknn)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (size_t knn = 1; knn < 20; knn += 3)
    {
        for (int r = 1; r < 5; r++)
        {
            const float rf = static_cast<float>(r);
            for (int i = 0; i < 100; i++)
            {
                rknn_L1_vs_bruteforce_test<float>(100, 2, knn, 9.0f * rf * rf);
                rknn_L1_vs_bruteforce_test<float>(100, 3, knn, 9.0f * rf * rf);
                rknn_L1_vs_bruteforce_test<float>(100, 7, knn, 9.0f * rf * rf);

                rknn_L1_vs_bruteforce_test<double>(100, 2, knn, 9.0 * r * r);
                rknn_L1_vs_bruteforce_test<double>(100, 3, knn, 9.0 * r * r);
                rknn_L1_vs_bruteforce_test<double>(100, 7, knn, 9.0 * r * r);
            }
        }
    }
}

TEST(kdtree, L2_vs_L2_simple)
{
    for (int nResults = 1; nResults < 10; nResults++)
    {
        L2_vs_L2_simple_test<float>(5, nResults);

        L2_vs_L2_simple_test<float>(100, nResults);
        L2_vs_L2_simple_test<double>(100, nResults);
    }
}

TEST(kdtree, robust_empty_tree)
{
    // Try to build a tree with 0 data points, to test
    // robustness against this situation:
    PointCloud<double> cloud;

    double query_pt[3] = {0.5, 0.5, 0.5};

    // construct a kd-tree index:
    typedef KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3 /* dim */
        >
        my_kd_tree_simple_t;

    my_kd_tree_simple_t index1(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    // Now we will try to search in the tree, and WE EXPECT a result of
    // no neighbors found if the error detection works fine:
    const size_t                    num_results = 1;
    std::vector<size_t>             ret_index(num_results);
    std::vector<double>             out_dist_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);
    bool result = index1.findNeighbors(resultSet, &query_pt[0]);
    EXPECT_EQ(result, false);
}

TEST(kdtree, L2_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int knn = 1; knn < 20; knn += 3)
    {
        for (int i = 0; i < 500; i++)
        {
            L2_vs_bruteforce_test<float>(10, 2, knn);

            L2_vs_bruteforce_test<float>(100, 2, knn);
            L2_vs_bruteforce_test<float>(100, 3, knn);
            L2_vs_bruteforce_test<float>(100, 7, knn);

            L2_vs_bruteforce_test<double>(100, 2, knn);
            L2_vs_bruteforce_test<double>(100, 3, knn);
            L2_vs_bruteforce_test<double>(100, 7, knn);
        }
    }
}

TEST(kdtree, box_L2_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 500; i++)
    {
        box_L2_vs_bruteforce_test<float>(10, 2);

        box_L2_vs_bruteforce_test<float>(100, 2);
        box_L2_vs_bruteforce_test<float>(100, 3);
        box_L2_vs_bruteforce_test<float>(100, 7);

        box_L2_vs_bruteforce_test<double>(100, 2);
        box_L2_vs_bruteforce_test<double>(100, 3);
        box_L2_vs_bruteforce_test<double>(100, 7);
    }
}

TEST(kdtree, L2_vs_bruteforce_rknn)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (size_t knn = 1; knn < 20; knn += 3)
    {
        for (int r = 1; r < 5; r++)
        {
            const float rf = static_cast<float>(r);
            for (int i = 0; i < 100; i++)
            {
                rknn_L2_vs_bruteforce_test<float>(100, 2, knn, 9.0f * rf * rf);
                rknn_L2_vs_bruteforce_test<float>(100, 3, knn, 9.0f * rf * rf);
                rknn_L2_vs_bruteforce_test<float>(100, 7, knn, 9.0f * rf * rf);

                rknn_L2_vs_bruteforce_test<double>(100, 2, knn, 9.0 * r * r);
                rknn_L2_vs_bruteforce_test<double>(100, 3, knn, 9.0 * r * r);
                rknn_L2_vs_bruteforce_test<double>(100, 7, knn, 9.0 * r * r);
            }
        }
    }
}

TEST(kdtree, SO3_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        SO3_vs_bruteforce_test<float>(5);

        SO3_vs_bruteforce_test<float>(100);
        SO3_vs_bruteforce_test<float>(100);
        SO3_vs_bruteforce_test<float>(100);

        SO3_vs_bruteforce_test<double>(100);
        SO3_vs_bruteforce_test<double>(100);
        SO3_vs_bruteforce_test<double>(100);
    }
}

TEST(kdtree, SO2_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        SO2_vs_bruteforce_test<float>(100);
        SO2_vs_bruteforce_test<float>(100);
        SO2_vs_bruteforce_test<float>(100);

        SO2_vs_bruteforce_test<double>(100);
        SO2_vs_bruteforce_test<double>(100);
        SO2_vs_bruteforce_test<double>(100);
    }
}

TEST(kdtree, L2_dynamic_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        L2_dynamic_vs_bruteforce_test<float>(100);
        L2_dynamic_vs_bruteforce_test<float>(100);
        L2_dynamic_vs_bruteforce_test<float>(100);

        L2_dynamic_vs_bruteforce_test<double>(100);
        L2_dynamic_vs_bruteforce_test<double>(100);
        L2_dynamic_vs_bruteforce_test<double>(100);
    }
}

TEST(kdtree, robust_nonempty_tree)
{
    // Try to build a dynamic tree with some initial points
    PointCloud<double> cloud;
    const size_t       max_point_count = 1000;
    generateRandomPointCloud(cloud, max_point_count);

    const double query_pt[3] = {0.5, 0.5, 0.5};

    // construct a kd-tree index:
    typedef KDTreeSingleIndexDynamicAdaptor<
        L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3>
        my_kd_tree_simple_t;

    my_kd_tree_simple_t index1(
        3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */), max_point_count);

    // Try a search and expect a neighbor to exist because the dynamic tree was
    // passed a non-empty cloud
    const size_t                    num_results = 1;
    std::vector<size_t>             ret_index(num_results);
    std::vector<double>             out_dist_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);
    bool result = index1.findNeighbors(resultSet, &query_pt[0]);
    EXPECT_EQ(result, true);
}

TEST(kdtree, add_and_remove_points)
{
    PointCloud<double> cloud;
    cloud.pts = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.7, 0.7, 0.7}};

    typedef KDTreeSingleIndexDynamicAdaptor<
        L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3>
        my_kd_tree_simple_t;

    my_kd_tree_simple_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));

    const auto query = [&index]() -> size_t
    {
        const double                    query_pt[3] = {0.5, 0.5, 0.5};
        const size_t                    num_results = 1;
        std::vector<size_t>             ret_index(num_results);
        std::vector<double>             out_dist_sqr(num_results);
        nanoflann::KNNResultSet<double> resultSet(num_results);

        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        index.findNeighbors(resultSet, &query_pt[0]);

        return ret_index[0];
    };

    auto actual = query();
    EXPECT_EQ(actual, static_cast<size_t>(1));

    index.removePoint(1);
    actual = query();
    EXPECT_EQ(actual, static_cast<size_t>(2));

    index.addPoints(1, 1);
    actual = query();
    EXPECT_EQ(actual, static_cast<size_t>(1));

    index.removePoint(1);
    index.removePoint(2);
    actual = query();
    EXPECT_EQ(actual, static_cast<size_t>(0));
}

// Test that repeated removePoint/addPoints cycles work correctly.
// This exercises the case where the dynamic adaptor's internal pointCount_
// grows beyond kdtree_get_point_count() due to re-adding removed points,
// which must not cause assertion failures in divideTree.
TEST(kdtree, dynamic_repeated_remove_readd)
{
    PointCloud<double> cloud;
    // Create enough points that sub-trees accumulate entries exceeding N
    const size_t N = 20;
    for (size_t i = 0; i < N; i++)
    {
        cloud.pts.push_back({
            static_cast<double>(i),
            static_cast<double>(i * 2),
            static_cast<double>(i * 3),
        });
    }

    typedef KDTreeSingleIndexDynamicAdaptor<
        L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3>
        my_kd_tree_t;

    my_kd_tree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));

    // Repeatedly remove and re-add groups of points, simulating the pattern
    // of temporarily excluding points from queries then restoring them.
    for (size_t iteration = 0; iteration < 10; iteration++)
    {
        // Remove a group of points
        for (size_t i = 0; i < N / 2; i++)
        {
            index.removePoint(i);
        }

        // Query while points are removed
        const double                    query_pt[3] = {5.0, 10.0, 15.0};
        std::vector<size_t>             ret_index(1);
        std::vector<double>             out_dist_sqr(1);
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        index.findNeighbors(resultSet, &query_pt[0]);

        // The nearest point should be among the non-removed ones (indices N/2..N-1)
        EXPECT_GE(ret_index[0], N / 2);

        // Re-add the removed points
        for (size_t i = 0; i < N / 2; i++)
        {
            index.addPoints(static_cast<uint32_t>(i), static_cast<uint32_t>(i));
        }

        // Query again - nearest should be index 5 (closest to query point)
        nanoflann::KNNResultSet<double> resultSet2(1);
        resultSet2.init(&ret_index[0], &out_dist_sqr[0]);
        index.findNeighbors(resultSet2, &query_pt[0]);
        EXPECT_EQ(ret_index[0], static_cast<size_t>(5));
    }
}

// Re-adding a previously-removed point must reactivate the existing entry in
// place, not insert a duplicate. Otherwise repeated remove/add cycles make the
// internal index vectors grow without bound and cause the same point index to
// be reported multiple times in radius/knn searches.
TEST(kdtree, dynamic_readd_no_duplicates)
{
    PointCloud<double> cloud;
    const size_t       N = 20;
    for (size_t i = 0; i < N; i++)
    {
        cloud.pts.push_back({
            static_cast<double>(i),
            static_cast<double>(i * 2),
            static_cast<double>(i * 3),
        });
    }

    typedef KDTreeSingleIndexDynamicAdaptor<
        L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3>
        my_kd_tree_t;

    my_kd_tree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));

    // Helper: total number of entries physically stored across all sub-trees.
    const auto total_stored_entries = [&index]() -> size_t
    {
        size_t total = 0;
        for (const auto& subtree : index.getAllIndices()) total += subtree.vAcc_.size();
        return total;
    };

    // Right after construction every point is stored exactly once.
    EXPECT_EQ(total_stored_entries(), N);

    // Repeatedly remove and re-add the same group of points.
    for (size_t iteration = 0; iteration < 10; iteration++)
    {
        for (size_t i = 0; i < N / 2; i++) index.removePoint(i);
        for (size_t i = 0; i < N / 2; i++)
            index.addPoints(static_cast<uint32_t>(i), static_cast<uint32_t>(i));
    }

    // The number of stored entries must not have grown: re-adding a removed
    // point reuses its slot rather than appending a duplicate.
    EXPECT_EQ(total_stored_entries(), N);

    // A radius search large enough to cover every point must return each point
    // index exactly once (no duplicates).
    const double                                       query_pt[3] = {0.0, 0.0, 0.0};
    std::vector<nanoflann::ResultItem<size_t, double>> matches;
    nanoflann::RadiusResultSet<double, size_t>         resultSet(1e30, matches);
    index.findNeighbors(resultSet, &query_pt[0]);

    EXPECT_EQ(matches.size(), N);
    std::set<size_t> unique_indices;
    for (const auto& m : matches) unique_indices.insert(m.first);
    EXPECT_EQ(unique_indices.size(), N);
}

TEST(kdtree, L2_concurrent_build_vs_bruteforce)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        L2_concurrent_build_vs_bruteforce_test<float>(100, 2);
        L2_concurrent_build_vs_bruteforce_test<float>(100, 3);
        L2_concurrent_build_vs_bruteforce_test<float>(100, 7);

        L2_concurrent_build_vs_bruteforce_test<double>(100, 2);
        L2_concurrent_build_vs_bruteforce_test<double>(100, 3);
        L2_concurrent_build_vs_bruteforce_test<double>(100, 7);
    }
}

TEST(kdtree, L2_concurrent_build_vs_L2)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < 10; i++)
    {
        L2_concurrent_build_vs_L2_test<float>(100, 2);
        L2_concurrent_build_vs_L2_test<float>(100, 3);
        L2_concurrent_build_vs_L2_test<float>(100, 7);

        L2_concurrent_build_vs_L2_test<double>(100, 2);
        L2_concurrent_build_vs_L2_test<double>(100, 3);
        L2_concurrent_build_vs_L2_test<double>(100, 7);
    }
}

TEST(kdtree, L2_static_vs_dynamic)
{
    for (int nResults = 0; nResults < 10; nResults++)
    {
        L2_dynamic_sorted_test<float>(100, nResults);
        L2_dynamic_sorted_test<double>(100, nResults);
    }
}

TEST(kdtree, same_points)
{
    using num_t         = double;
    using point_cloud_t = PointCloud<num_t>;
    using kdtree_t      = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<num_t, point_cloud_t>, point_cloud_t, 3 /* dim */>;

    point_cloud_t cloud;
    cloud.pts.resize(16);
    for (size_t i = 0; i < 16; ++i)
    {
        cloud.pts[i].x = -1.;
        cloud.pts[i].y = 0.;
        cloud.pts[i].z = 1.;
    }

    kdtree_t idx(3 /*dim*/, cloud);
}

// Regression test: saveIndex/loadIndex for an empty dataset.
// saveIndex conditionally skips save_tree when root_node_==nullptr, but
// loadIndex used to call load_tree unconditionally, reading past EOF on the
// stream (setting its failbit) and leaving root_node_ as a non-null pointer to
// garbage memory.
TEST(kdtree, saveload_empty_index)
{
    using num_t    = double;
    using cloud_t  = PointCloud<num_t>;
    using kdtree_t = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<num_t, cloud_t>, cloud_t, 3>;

    cloud_t emptyCloud;  // 0 points

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    {
        const kdtree_t index(3, emptyCloud, KDTreeSingleIndexAdaptorParams(10));
        index.saveIndex(ss);
    }

    {
        kdtree_t index(
            3, emptyCloud,
            KDTreeSingleIndexAdaptorParams(
                10, KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex));
        index.loadIndex(ss);

        // Bug: load_tree always ran even for empty indices, reading sizeof(Node)
        // bytes past EOF and setting the stream's failbit.
        EXPECT_FALSE(ss.fail());

        num_t                          query_pt[3] = {0.5, 0.5, 0.5};
        nanoflann::KNNResultSet<num_t> resultSet(1);
        size_t                         ret_index;
        num_t                          out_dist_sqr;
        resultSet.init(&ret_index, &out_dist_sqr);
        EXPECT_FALSE(index.findNeighbors(resultSet, query_pt));
    }
}

// Verify that a non-empty index produces identical search results after a
// save/load round-trip.
TEST(kdtree, saveload_nonempty_index)
{
    srand(42);

    using num_t    = double;
    using cloud_t  = PointCloud<num_t>;
    using kdtree_t = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<num_t, cloud_t>, cloud_t, 3>;

    cloud_t cloud;
    generateRandomPointCloud(cloud, 10000);

    const num_t query_pt[3] = {0.5, 0.5, 0.5};
    size_t      expected_idx;
    num_t       expected_dist;

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    {
        kdtree_t                       index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
        nanoflann::KNNResultSet<num_t> rs(1);
        rs.init(&expected_idx, &expected_dist);
        index.findNeighbors(rs, query_pt);
        index.saveIndex(ss);
    }

    {
        kdtree_t index(
            3, cloud,
            KDTreeSingleIndexAdaptorParams(
                10, KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex));
        index.loadIndex(ss);

        size_t                         loaded_idx;
        num_t                          loaded_dist;
        nanoflann::KNNResultSet<num_t> rs(1);
        rs.init(&loaded_idx, &loaded_dist);
        index.findNeighbors(rs, query_pt);

        EXPECT_EQ(loaded_idx, expected_idx);
        EXPECT_NEAR(loaded_dist, expected_dist, 1e-10);
    }
}

// Regression test: saveIndex/loadIndex on KDTreeSingleIndexDynamicAdaptor_ used
// to recurse infinitely because the 1-arg method name-hid Base::saveIndex and
// called itself instead of Base::saveIndex(*this, stream).
TEST(kdtree, dynamic_saveload_roundtrip)
{
    srand(42);

    using num_t    = double;
    using cloud_t  = PointCloud<num_t>;
    using dist_t   = nanoflann::L2_Simple_Adaptor<num_t, cloud_t>;
    using subidx_t = KDTreeSingleIndexDynamicAdaptor_<dist_t, cloud_t, 3>;

    cloud_t cloud;
    generateRandomPointCloud(cloud, 200);

    // Build a sub-tree directly with all points assigned to tree slot 0.
    std::vector<int> treeIndex(cloud.pts.size(), 0);
    subidx_t         orig(3, cloud, treeIndex);
    for (size_t i = 0; i < cloud.pts.size(); ++i)
    {
        orig.vAcc_.push_back(static_cast<uint32_t>(i));
    }
    orig.buildIndex();

    const num_t query_pt[3] = {0.5, 0.5, 0.5};

    size_t                         expected_idx;
    num_t                          expected_dist;
    nanoflann::KNNResultSet<num_t> rs_orig(1);
    rs_orig.init(&expected_idx, &expected_dist);
    orig.findNeighbors(rs_orig, query_pt, nanoflann::SearchParameters());

    // Save the sub-tree index.
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    orig.saveIndex(ss);
    EXPECT_FALSE(ss.fail());

    // Load into a fresh sub-tree and verify the search result matches.
    subidx_t loaded(3, cloud, treeIndex);
    loaded.loadIndex(ss);
    EXPECT_FALSE(ss.fail());

    size_t                         loaded_idx;
    num_t                          loaded_dist;
    nanoflann::KNNResultSet<num_t> rs_loaded(1);
    rs_loaded.init(&loaded_idx, &loaded_dist);
    loaded.findNeighbors(rs_loaded, query_pt, nanoflann::SearchParameters());

    EXPECT_EQ(loaded_idx, expected_idx);
    EXPECT_NEAR(loaded_dist, expected_dist, 1e-10);
}

// loadIndex() must throw when the stream does not start with the nanoflann magic.
TEST(kdtree, saveload_bad_magic)
{
    using num_t    = double;
    using cloud_t  = PointCloud<num_t>;
    using kdtree_t = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<num_t, cloud_t>, cloud_t, 3>;

    cloud_t cloud;
    generateRandomPointCloud(cloud, 100);

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    {
        kdtree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
        index.saveIndex(ss);
    }

    // Overwrite the first four bytes (magic) with garbage.
    ss.seekp(0);
    const uint32_t bad_magic = 0xDEADBEEF;
    ss.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
    ss.seekg(0);

    kdtree_t index(
        3, cloud,
        KDTreeSingleIndexAdaptorParams(10, KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex));
    EXPECT_THROW(index.loadIndex(ss), std::runtime_error);
}

// loadIndex() must throw when the nanoflann version in the file differs.
TEST(kdtree, saveload_version_mismatch)
{
    using num_t    = double;
    using cloud_t  = PointCloud<num_t>;
    using kdtree_t = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<num_t, cloud_t>, cloud_t, 3>;

    cloud_t cloud;
    generateRandomPointCloud(cloud, 100);

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    {
        kdtree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
        index.saveIndex(ss);
    }

    // Overwrite the version field (bytes 4-7) with a fake version.
    ss.seekp(4);
    const uint32_t bad_version = 0x000;
    ss.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
    ss.seekg(0);

    kdtree_t index(
        3, cloud,
        KDTreeSingleIndexAdaptorParams(10, KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex));
    EXPECT_THROW(index.loadIndex(ss), std::runtime_error);
}

// ===========================================================================
//  Tests for KDTreeSingleIndexIncrementalAdaptor (single self-balancing tree)
// ===========================================================================
namespace
{
using inc_cloud_t = PointCloud<double>;
using inc_tree_t  = nanoflann::KDTreeSingleIndexIncrementalAdaptor<
    nanoflann::L2_Simple_Adaptor<double, inc_cloud_t>, inc_cloud_t, 3, uint32_t>;

// Brute-force nearest neighbor over an explicit set of "live" indices.
double bruteforce_nn(
    const inc_cloud_t& cloud, const std::set<uint32_t>& live, const double* q, uint32_t& bestIdx)
{
    double best = std::numeric_limits<double>::max();
    bestIdx     = 0;
    for (uint32_t i : live)
    {
        const double dx = q[0] - cloud.pts[i].x, dy = q[1] - cloud.pts[i].y,
                     dz = q[2] - cloud.pts[i].z;
        const double d  = dx * dx + dy * dy + dz * dz;
        if (d < best)
        {
            best    = d;
            bestIdx = i;
        }
    }
    return best;
}

bool inc_in_box(const inc_cloud_t& c, uint32_t i, const double lo[3], const double hi[3])
{
    return c.pts[i].x >= lo[0] && c.pts[i].x <= hi[0] && c.pts[i].y >= lo[1] &&
           c.pts[i].y <= hi[1] && c.pts[i].z >= lo[2] && c.pts[i].z <= hi[2];
}
}  // namespace

TEST(kdtree_incremental, add_knn_vs_bruteforce)
{
    srand(42);
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    std::set<uint32_t> live;

    // Insert in several batches (interleaved, like a streaming source).
    for (int batch = 0; batch < 6; ++batch)
    {
        const uint32_t start = static_cast<uint32_t>(cloud.pts.size());
        const size_t   add   = 300 + batch * 50;
        for (size_t i = 0; i < add; ++i)
            cloud.pts.push_back({10.0 * (rand() % 1000) / 1000.0,
                                 10.0 * (rand() % 1000) / 1000.0,
                                 10.0 * (rand() % 1000) / 1000.0});
        const uint32_t end = static_cast<uint32_t>(cloud.pts.size()) - 1;
        index.addPoints(start, end);
        for (uint32_t i = start; i <= end; ++i) live.insert(i);
    }
    EXPECT_EQ(index.size(), live.size());

    for (int t = 0; t < 300; ++t)
    {
        const double q[3] = {10.0 * (rand() % 1000) / 1000.0, 10.0 * (rand() % 1000) / 1000.0,
                             10.0 * (rand() % 1000) / 1000.0};
        uint32_t ri;
        double   rd;
        const size_t n = index.knnSearch(q, 1, &ri, &rd);
        uint32_t bi;
        const double bd = bruteforce_nn(cloud, live, q, bi);
        ASSERT_EQ(n, 1u);
        EXPECT_NEAR(rd, bd, 1e-9);
    }
}

TEST(kdtree_incremental, remove_knn_vs_bruteforce)
{
    srand(7);
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    const size_t N = 2000;
    for (size_t i = 0; i < N; ++i)
        cloud.pts.push_back({10.0 * (rand() % 1000) / 1000.0, 10.0 * (rand() % 1000) / 1000.0,
                             10.0 * (rand() % 1000) / 1000.0});
    index.addPoints(0, static_cast<uint32_t>(N) - 1);

    std::set<uint32_t> live;
    for (uint32_t i = 0; i < N; ++i) live.insert(i);

    // Remove ~40% of points at random.
    std::vector<uint32_t> order(live.begin(), live.end());
    std::mt19937          g(99);
    std::shuffle(order.begin(), order.end(), g);
    for (size_t i = 0; i < order.size() * 4 / 10; ++i)
    {
        index.removePoint(order[i]);
        live.erase(order[i]);
    }
    EXPECT_EQ(index.size(), live.size());

    // Removing twice / removing an out-of-range index must be a no-op.
    index.removePoint(order[0]);
    index.removePoint(1000000);
    EXPECT_EQ(index.size(), live.size());

    for (int t = 0; t < 300; ++t)
    {
        const double q[3] = {10.0 * (rand() % 1000) / 1000.0, 10.0 * (rand() % 1000) / 1000.0,
                             10.0 * (rand() % 1000) / 1000.0};
        uint32_t ri;
        double   rd;
        const size_t n = index.knnSearch(q, 1, &ri, &rd);
        uint32_t bi;
        const double bd = bruteforce_nn(cloud, live, q, bi);
        ASSERT_EQ(n, 1u);
        EXPECT_TRUE(live.count(ri) == 1);
        EXPECT_NEAR(rd, bd, 1e-9);
    }
}

TEST(kdtree_incremental, removeOutsideBox_vs_bruteforce)
{
    srand(123);
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    const size_t N = 3000;
    for (size_t i = 0; i < N; ++i)
        cloud.pts.push_back({20.0 * (rand() % 1000) / 1000.0 - 10.0,
                             20.0 * (rand() % 1000) / 1000.0 - 10.0,
                             20.0 * (rand() % 1000) / 1000.0 - 10.0});
    index.addPoints(0, static_cast<uint32_t>(N) - 1);

    const double lo[3] = {-3, -3, -3}, hi[3] = {3, 3, 3};
    inc_tree_t::BoundingBox keep;
    for (int d = 0; d < 3; ++d) { keep[d].low = lo[d]; keep[d].high = hi[d]; }
    index.removeOutsideBox(keep);

    std::set<uint32_t> live;
    for (uint32_t i = 0; i < N; ++i)
        if (inc_in_box(cloud, i, lo, hi)) live.insert(i);
    EXPECT_EQ(index.size(), live.size());

    for (int t = 0; t < 300; ++t)
    {
        const double q[3] = {6.0 * (rand() % 1000) / 1000.0 - 3.0,
                             6.0 * (rand() % 1000) / 1000.0 - 3.0,
                             6.0 * (rand() % 1000) / 1000.0 - 3.0};
        uint32_t ri;
        double   rd;
        const size_t n = index.knnSearch(q, 1, &ri, &rd);
        ASSERT_EQ(n, 1u);
        EXPECT_TRUE(live.count(ri) == 1);
        uint32_t bi;
        const double bd = bruteforce_nn(cloud, live, q, bi);
        EXPECT_NEAR(rd, bd, 1e-9);
    }
}

TEST(kdtree_incremental, removeBox_and_findWithinBox)
{
    srand(321);
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    const size_t N = 3000;
    for (size_t i = 0; i < N; ++i)
        cloud.pts.push_back({20.0 * (rand() % 1000) / 1000.0 - 10.0,
                             20.0 * (rand() % 1000) / 1000.0 - 10.0,
                             20.0 * (rand() % 1000) / 1000.0 - 10.0});
    index.addPoints(0, static_cast<uint32_t>(N) - 1);
    std::set<uint32_t> live;
    for (uint32_t i = 0; i < N; ++i) live.insert(i);

    // Delete an inner box.
    const double blo[3] = {-2, -2, -2}, bhi[3] = {2, 2, 2};
    inc_tree_t::BoundingBox box;
    for (int d = 0; d < 3; ++d) { box[d].low = blo[d]; box[d].high = bhi[d]; }
    index.removeBox(box);
    for (auto it = live.begin(); it != live.end();)
    {
        if (inc_in_box(cloud, *it, blo, bhi)) it = live.erase(it);
        else ++it;
    }
    EXPECT_EQ(index.size(), live.size());

    // findWithinBox over a different query box, compared against the oracle.
    const double qlo[3] = {-5, -5, -5}, qhi[3] = {5, 5, 5};
    inc_tree_t::BoundingBox qbox;
    for (int d = 0; d < 3; ++d) { qbox[d].low = qlo[d]; qbox[d].high = qhi[d]; }
    std::vector<uint32_t>            got;
    nanoflann::BoxResultSet<uint32_t> brs(got);
    (void)index.findWithinBox(brs, qbox);

    std::set<uint32_t> expected;
    for (uint32_t i : live)
        if (inc_in_box(cloud, i, qlo, qhi)) expected.insert(i);
    EXPECT_EQ(got.size(), expected.size());
    for (uint32_t i : got) EXPECT_TRUE(expected.count(i) == 1);
}

TEST(kdtree_incremental, bounded_memory_under_churn)
{
    // Sliding-window LiDAR-like churn: constant-velocity insert + trim. The
    // physical node count (live + tombstones) must stay bounded, not grow with
    // the total number of inserted points.
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    std::mt19937 g(2024);
    std::uniform_real_distribution<double> U(0.0, 10.0);
    std::set<uint32_t> live;

    double  cx      = 0.0;
    size_t  maxPhys = 0;
    for (int frame = 0; frame < 150; ++frame)
    {
        cx += 1.0;  // constant velocity
        const uint32_t start = static_cast<uint32_t>(cloud.pts.size());
        for (int i = 0; i < 400; ++i) cloud.pts.push_back({cx + U(g), U(g), U(g)});
        const uint32_t end = static_cast<uint32_t>(cloud.pts.size()) - 1;
        index.addPoints(start, end);
        for (uint32_t i = start; i <= end; ++i) live.insert(i);

        inc_tree_t::BoundingBox keep;
        keep[0].low = cx - 5; keep[0].high = cx + 15;
        keep[1].low = -1e9;   keep[1].high = 1e9;
        keep[2].low = -1e9;   keep[2].high = 1e9;
        index.removeOutsideBox(keep);
        for (auto it = live.begin(); it != live.end();)
        {
            const double x = cloud.pts[*it].x;
            if (x < cx - 5 || x > cx + 15) it = live.erase(it);
            else ++it;
        }
        ASSERT_EQ(index.size(), live.size());
        if (frame > 30) maxPhys = std::max(maxPhys, index.physicalSize());
    }
    // Steady-state window holds ~8000 live points; tombstones must stay a small
    // multiple of that, NOT proportional to the ~60000 total inserts.
    EXPECT_LT(index.physicalSize(), 4 * index.size());
    EXPECT_LT(maxPhys, static_cast<size_t>(30000));
}

TEST(kdtree_incremental, radius_and_rknn_vs_bruteforce)
{
    srand(555);
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    const size_t N = 1500;
    for (size_t i = 0; i < N; ++i)
        cloud.pts.push_back({10.0 * (rand() % 1000) / 1000.0, 10.0 * (rand() % 1000) / 1000.0,
                             10.0 * (rand() % 1000) / 1000.0});
    index.addPoints(0, static_cast<uint32_t>(N) - 1);
    std::set<uint32_t> live;
    for (uint32_t i = 0; i < N; ++i) live.insert(i);

    const double radius = 1.5;  // squared-distance threshold for L2
    for (int t = 0; t < 100; ++t)
    {
        const double q[3] = {10.0 * (rand() % 1000) / 1000.0, 10.0 * (rand() % 1000) / 1000.0,
                             10.0 * (rand() % 1000) / 1000.0};
        std::vector<nanoflann::ResultItem<uint32_t, double>> matches;
        (void)index.radiusSearch(q, radius, matches);

        size_t expected = 0;
        for (uint32_t i : live)
        {
            const double dx = q[0] - cloud.pts[i].x, dy = q[1] - cloud.pts[i].y,
                         dz = q[2] - cloud.pts[i].z;
            if (dx * dx + dy * dy + dz * dz < radius) expected++;
        }
        EXPECT_EQ(matches.size(), expected);
    }
}

// Reclamation contract: after removeOutsideBox + the rebuild it triggers,
// acquireRemovedPoints() must report exactly the physically-evicted indices
// (all outside the keep box, unique) so the caller can recycle those dataset
// slots and keep its storage bounded.
TEST(kdtree_incremental, removeOutsideBox_acquireRemovedPoints)
{
    srand(99);
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    index.setCollectRemovedPoints(true);
    const size_t N = 4000;
    for (size_t i = 0; i < N; ++i)
        cloud.pts.push_back({20.0 * (rand() % 1000) / 1000.0 - 10.0,
                             20.0 * (rand() % 1000) / 1000.0 - 10.0,
                             20.0 * (rand() % 1000) / 1000.0 - 10.0});
    index.addPoints(0, static_cast<uint32_t>(N) - 1);

    const double lo[3] = {-4, -4, -4}, hi[3] = {4, 4, 4};
    inc_tree_t::BoundingBox keep;
    for (int d = 0; d < 3; ++d) { keep[d].low = lo[d]; keep[d].high = hi[d]; }
    index.removeOutsideBox(keep);

    std::set<uint32_t> expectedEvicted, expectedLive;
    for (uint32_t i = 0; i < N; ++i)
        (inc_in_box(cloud, i, lo, hi) ? expectedLive : expectedEvicted).insert(i);

    const std::vector<uint32_t> reported = index.acquireRemovedPoints();
    const std::set<uint32_t>    reportedSet(reported.begin(), reported.end());
    EXPECT_EQ(reported.size(), reportedSet.size());  // unique
    // Every reported slot is genuinely evicted (outside) and none is still live.
    for (uint32_t i : reportedSet)
    {
        EXPECT_TRUE(expectedEvicted.count(i) == 1);
        EXPECT_TRUE(expectedLive.count(i) == 0);
    }
    // A fully reclaimed run (heavy deletion -> root rebuild) reports them all.
    if (index.physicalSize() == index.size())
        EXPECT_EQ(reportedSet, expectedEvicted);
    // A second call returns nothing new.
    EXPECT_TRUE(index.acquireRemovedPoints().empty());
}

// boundingBox(), reserve(), and buildFromIndices() on a *populated* index
// (rebuild-from-subset; exercises the recycle-existing-nodes branch).
TEST(kdtree_incremental, buildFromIndices_boundingBox_reserve)
{
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    index.reserve(1000);  // pre-size the index->node map
    const size_t N = 600;
    for (size_t i = 0; i < N; ++i)
        cloud.pts.push_back({static_cast<double>(i % 10), static_cast<double>((i / 10) % 10),
                             static_cast<double>(i / 100)});
    index.addPoints(0, static_cast<uint32_t>(N) - 1);

    // boundingBox encloses all inserted points.
    auto bb = index.boundingBox();
    EXPECT_LE(bb[0].low, 0.0);
    EXPECT_GE(bb[0].high, 9.0);
    EXPECT_GE(bb[2].high, 5.0);

    // Rebuild from a subset of indices (this index is non-empty -> takes the
    // recycle-existing-nodes path inside buildFromIndices).
    std::vector<uint32_t> subset;
    for (uint32_t i = 0; i < N; i += 2) subset.push_back(i);  // keep even indices
    index.buildFromIndices(subset);
    EXPECT_EQ(index.size(), subset.size());
    EXPECT_EQ(index.physicalSize(), subset.size());  // fresh balanced tree, no tombstones

    // KNN now only returns kept (even) indices.
    std::set<uint32_t> kept(subset.begin(), subset.end());
    for (uint32_t i = 0; i < N; ++i)
    {
        const double q[3] = {cloud.pts[i].x, cloud.pts[i].y, cloud.pts[i].z};
        uint32_t     ri;
        double       rd;
        ASSERT_EQ(index.knnSearch(q, 1, &ri, &rd), 1u);
        EXPECT_TRUE(kept.count(ri) == 1);
    }
}

// Insert into a region whose subtree was bulk-killed by removeOutsideBox while
// inline rebuilds are disabled, so the lazy whole-subtree tombstone survives and
// the insertion descent must push it down (covers pushDownDelete()).
TEST(kdtree_incremental, pushdown_delete_into_killed_region)
{
    inc_cloud_t cloud;
    inc_tree_t  index(3, cloud);
    index.setInlineRebuild(false);  // keep treeDeleted subtrees around (no reclaim)

    // Two well-separated clusters so they live in separate subtrees.
    for (int i = 0; i < 200; ++i) cloud.pts.push_back({0.1 * i / 200.0, 0.0, 0.0});  // A near x=0
    for (int i = 0; i < 200; ++i)
        cloud.pts.push_back({100.0 + 0.1 * i / 200.0, 0.0, 0.0});  // B near x=100
    index.addPoints(0, 399);
    ASSERT_EQ(index.size(), 400u);

    // Kill cluster B (everything with x>10): its subtree becomes treeDeleted and,
    // with inline rebuild off, is NOT reclaimed.
    inc_tree_t::BoundingBox keep;
    keep[0].low = -1; keep[0].high = 10;
    keep[1].low = -1; keep[1].high = 1;
    keep[2].low = -1; keep[2].high = 1;
    index.removeOutsideBox(keep);
    ASSERT_EQ(index.size(), 200u);  // only cluster A remains live

    // Now insert a fresh point inside the killed region (x~100): the descent
    // reaches the treeDeleted subtree and must push the deletion down.
    const uint32_t newIdx = static_cast<uint32_t>(cloud.pts.size());
    cloud.pts.push_back({100.05, 0.0, 0.0});
    index.addPoint(newIdx);
    EXPECT_EQ(index.size(), 201u);

    // The new point is the nearest neighbor of its own location; the old killed
    // points stay deleted.
    const double q[3] = {100.05, 0.0, 0.0};
    uint32_t     ri;
    double       rd;
    ASSERT_EQ(index.knnSearch(q, 1, &ri, &rd), 1u);
    EXPECT_EQ(ri, newIdx);
    EXPECT_NEAR(rd, 0.0, 1e-9);
}

#ifndef NANOFLANN_NO_THREADS
TEST(kdtree_incremental, async_mt_vs_bruteforce_under_churn)
{
    using mt_tree_t = nanoflann::KDTreeSingleIndexIncrementalAdaptorMT<
        nanoflann::L2_Simple_Adaptor<double, inc_cloud_t>, inc_cloud_t, 3, uint32_t>;

    inc_cloud_t cloud;
    cloud.pts.reserve(400000);  // stable storage required while a rebuild is in flight
    mt_tree_t   index(3, cloud, nanoflann::KDTreeIncrementalIndexParams(0.8f, 0.5f), 1.3, 2000);
    std::set<uint32_t> live;
    std::mt19937       g(2024);
    std::uniform_real_distribution<double> U(0.0, 10.0);

    double cx = 0.0;
    bool   sawRebuild = false;
    for (int frame = 0; frame < 80; ++frame)
    {
        cx += 1.0;
        const uint32_t start = static_cast<uint32_t>(cloud.pts.size());
        for (int i = 0; i < 1500; ++i) cloud.pts.push_back({cx + U(g), U(g), U(g)});
        const uint32_t end = static_cast<uint32_t>(cloud.pts.size()) - 1;
        index.addPoints(start, end);
        for (uint32_t i = start; i <= end; ++i) live.insert(i);

        mt_tree_t::BoundingBox keep;
        keep[0].low = cx - 5; keep[0].high = cx + 15;
        keep[1].low = -1e9;   keep[1].high = 1e9;
        keep[2].low = -1e9;   keep[2].high = 1e9;
        index.removeOutsideBox(keep);
        for (auto it = live.begin(); it != live.end();)
        {
            const double x = cloud.pts[*it].x;
            if (x < cx - 5 || x > cx + 15) it = live.erase(it);
            else ++it;
        }
        if (index.isRebuilding()) sawRebuild = true;
        ASSERT_EQ(index.size(), live.size());
    }
    index.sync();  // finish any in-flight background rebuild
    EXPECT_TRUE(sawRebuild);  // background rebuilds actually triggered
    EXPECT_EQ(index.size(), live.size());
    EXPECT_LT(index.physicalSize(), 4 * index.size());  // bounded memory

    for (int t = 0; t < 500; ++t)
    {
        const double q[3] = {cx + U(g), U(g), U(g)};
        uint32_t ri;
        double   rd;
        const size_t n = index.knnSearch(q, 1, &ri, &rd);
        ASSERT_EQ(n, 1u);
        EXPECT_TRUE(live.count(ri) == 1);
        uint32_t bi;
        const double bd = bruteforce_nn(cloud, live, q, bi);
        EXPECT_NEAR(rd, bd, 1e-9);
    }
}

// The MT index must also report freed dataset slots (across background rebuilds)
// so a streaming caller can recycle storage and stay bounded.
TEST(kdtree_incremental, async_mt_slot_recycling_bounds_dataset)
{
    using mt_tree_t = nanoflann::KDTreeSingleIndexIncrementalAdaptorMT<
        nanoflann::L2_Simple_Adaptor<double, inc_cloud_t>, inc_cloud_t, 3, uint32_t>;

    inc_cloud_t cloud;
    cloud.pts.reserve(60000);  // bounded thanks to recycling; reserve for MT safety
    mt_tree_t   index(3, cloud, nanoflann::KDTreeIncrementalIndexParams(0.8f, 0.5f), 1.3, 2000);
    index.setCollectRemovedPoints(true);

    std::vector<uint32_t> freeSlots;
    std::set<uint32_t>    live;
    std::mt19937          g(2024);
    std::uniform_real_distribution<double> U(0.0, 10.0);

    double cx = 0.0;
    size_t totalInserted = 0, maxDataset = 0;
    for (int frame = 0; frame < 120; ++frame)
    {
        cx += 1.0;
        for (int i = 0; i < 800; ++i)
        {
            const inc_cloud_t::Point p{cx + U(g), U(g), U(g)};
            uint32_t slot;
            if (!freeSlots.empty()) { slot = freeSlots.back(); freeSlots.pop_back(); cloud.pts[slot] = p; }
            else { slot = static_cast<uint32_t>(cloud.pts.size()); cloud.pts.push_back(p); }
            index.addPoint(slot);
            live.insert(slot);
            ++totalInserted;
        }
        mt_tree_t::BoundingBox keep;
        keep[0].low = cx - 5; keep[0].high = cx + 15;
        keep[1].low = -1e9;   keep[1].high = 1e9;
        keep[2].low = -1e9;   keep[2].high = 1e9;
        index.removeOutsideBox(keep);
        for (auto it = live.begin(); it != live.end();)
        {
            const double x = cloud.pts[*it].x;
            if (x < cx - 5 || x > cx + 15) it = live.erase(it);
            else ++it;
        }
        for (uint32_t s : index.acquireRemovedPoints()) freeSlots.push_back(s);
        maxDataset = std::max(maxDataset, cloud.pts.size());
        ASSERT_EQ(index.size(), live.size());
    }
    index.sync();
    for (uint32_t s : index.acquireRemovedPoints()) freeSlots.push_back(s);

    // Dataset stayed bounded (far below total inserts) via slot recycling.
    EXPECT_LT(maxDataset, totalInserted / 2);

    // Final KNN correctness on recycled storage.
    for (int t = 0; t < 300; ++t)
    {
        const double q[3] = {cx + U(g), U(g), U(g)};
        uint32_t ri;
        double   rd;
        const size_t n = index.knnSearch(q, 1, &ri, &rd);
        ASSERT_EQ(n, 1u);
        EXPECT_TRUE(live.count(ri) == 1);
        uint32_t bi;
        const double bd = bruteforce_nn(cloud, live, q, bi);
        EXPECT_NEAR(rd, bd, 1e-9);
    }
}

// Exercises the MT wrapper's op-log replay for Remove and RemoveBox (ops issued
// while a background rebuild is in flight), the background rebuild callback, and
// the forwarder observers (boundingBox / snapshotLiveIndices / reserve).
TEST(kdtree_incremental, async_mt_replay_remove_box_and_callback)
{
    using mt_tree_t = nanoflann::KDTreeSingleIndexIncrementalAdaptorMT<
        nanoflann::L2_Simple_Adaptor<double, inc_cloud_t>, inc_cloud_t, 3, uint32_t>;

    inc_cloud_t cloud;
    cloud.pts.reserve(600000);  // stable storage for the background reads
    mt_tree_t   index(3, cloud, nanoflann::KDTreeIncrementalIndexParams(0.8f, 0.5f), 1.3, 2000);
    index.reserve(600000);

    // The callback runs on the background worker thread for each rebuilt tree.
    std::atomic<int> cbCalls{0};
    index.setRebuildCallback(
        [&cbCalls](mt_tree_t::Inner& fresh)
        {
            // Touch the fresh tree to make sure it is usable inside the callback.
            (void)fresh.size();
            cbCalls.fetch_add(1, std::memory_order_relaxed);
        });

    std::set<uint32_t> live;
    std::mt19937       g(123);
    std::uniform_real_distribution<double> U(-50.0, 50.0);

    auto inBox = [](const inc_cloud_t& c, uint32_t i, const double lo[3], const double hi[3])
    { return inc_in_box(c, i, lo, hi); };

    // Several rounds: each big insert re-triggers a background rebuild, and the
    // removePoint / removeBox issued right afterwards land in the op-log while
    // the (large) build is still running.
    for (int round = 0; round < 4; ++round)
    {
        const uint32_t start = static_cast<uint32_t>(cloud.pts.size());
        for (int i = 0; i < 60000; ++i) cloud.pts.push_back({U(g), U(g), U(g)});
        const uint32_t end = static_cast<uint32_t>(cloud.pts.size()) - 1;
        index.addPoints(start, end);  // triggers a background rebuild
        for (uint32_t i = start; i <= end; ++i) live.insert(i);

        // Remove a handful of individual points (logged Remove while building).
        for (uint32_t k = 0; k < 20; ++k)
        {
            const uint32_t victim = start + k * 7;
            index.removePoint(victim);
            live.erase(victim);
        }
        // Remove a box region (logged RemoveBox while building).
        const double blo[3] = {-50, -50, -50}, bhi[3] = {-30, -30, -30};
        mt_tree_t::BoundingBox box;
        for (int d = 0; d < 3; ++d) { box[d].low = blo[d]; box[d].high = bhi[d]; }
        index.removeBox(box);
        for (auto it = live.begin(); it != live.end();)
        {
            if (inBox(cloud, *it, blo, bhi)) it = live.erase(it);
            else ++it;
        }
    }
    index.sync();  // drain the final background rebuild (replays the op-log)

    EXPECT_GT(cbCalls.load(), 0);  // background callback actually ran
    EXPECT_EQ(index.size(), live.size());

    // Forwarder observers:
    EXPECT_EQ(index.physicalSize() >= index.size(), true);
    std::vector<uint32_t> liveIdx;
    index.snapshotLiveIndices(liveIdx);
    EXPECT_EQ(liveIdx.size(), live.size());
    auto bb = index.boundingBox();
    EXPECT_LE(bb[0].low, bb[0].high);

    // Correctness vs brute force on the final live set.
    for (int t = 0; t < 500; ++t)
    {
        const double q[3] = {U(g), U(g), U(g)};
        uint32_t     ri;
        double       rd;
        const size_t n = index.knnSearch(q, 1, &ri, &rd);
        ASSERT_EQ(n, 1u);
        EXPECT_TRUE(live.count(ri) == 1);
        uint32_t     bi;
        const double bd = bruteforce_nn(cloud, live, q, bi);
        EXPECT_NEAR(rd, bd, 1e-9);
    }
}
#endif  // NANOFLANN_NO_THREADS
