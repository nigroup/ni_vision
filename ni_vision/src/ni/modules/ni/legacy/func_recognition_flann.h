/** @file Functions for computing the flann, copied from the FLANN library
 */
#ifndef _NI_LEGACY_FUNC_RECOGNITION_FLANN_H_
#define _NI_LEGACY_FUNC_RECOGNITION_FLANN_H_

#include "flann/flann.h"

static flann_distance_t flann_distance_type = FLANN_DIST_EUCLIDEAN;

void flann_log_verbosity(int level);

void init_flann_parameters(FLANNParameters* p);

flann::IndexParams create_parameters(FLANNParameters* p);

template<typename Distance>
flann_index_t __flann_build_index(typename Distance::ElementType* dataset, int rows, int cols, float* speedup,
                                  FLANNParameters* flann_params, Distance d = Distance())
{
    typedef typename Distance::ElementType ElementType;
    try {

        init_flann_parameters(flann_params);
        if (flann_params == NULL) {
            throw FLANNException("The flann_params argument must be non-null");
        }
        IndexParams params = create_parameters(flann_params);
        Index<Distance>* index = new Index<Distance>(Matrix<ElementType>(dataset,rows,cols), params, d);
        index->buildIndex();
        params = index->getParameters();

        // FIXME
        //index_params->toParameters(*flann_params);

//        if (index->getType()==FLANN_INDEX_AUTOTUNED) {
//            AutotunedIndex<Distance>* autotuned_index = (AutotunedIndex<Distance>*)index->getIndex();
//            // FIXME
//            flann_params->checks = autotuned_index->getSearchParameters().checks;
//            *speedup = autotuned_index->getSpeedup();
//        }

        return index;
    }
    catch (std::runtime_error& e) {
        Logger::error("Caught exception: %s\n",e.what());
        return NULL;
    }
}


template<typename T>
flann_index_t _flann_build_index(T* dataset, int rows, int cols, float* speedup, FLANNParameters* flann_params)
{
    if (flann_distance_type==FLANN_DIST_EUCLIDEAN) {
        return __flann_build_index<L2<T> >(dataset, rows, cols, speedup, flann_params);
    }
    else {
        Logger::error( "Distance type unsupported in the C bindings, use the C++ bindings instead\n");
        return NULL;
    }
}

flann_index_t flann_build_index(float* dataset, int rows, int cols, float* speedup, FLANNParameters* flann_params);

template<typename Distance>
int __flann_find_nearest_neighbors_index(flann_index_t index_ptr, typename Distance::ElementType* testset, int tcount,
                                         int* result, typename Distance::ResultType* dists, int nn, FLANNParameters* flann_params)
{
    typedef typename Distance::ElementType ElementType;
    typedef typename Distance::ResultType DistanceType;

    try {
        init_flann_parameters(flann_params);
        if (index_ptr==NULL) {
            throw FLANNException("Invalid index");
        }
        Index<Distance>* index = (Index<Distance>*)index_ptr;

        Matrix<int> m_indices(result,tcount, nn);
        Matrix<DistanceType> m_dists(dists, tcount, nn);

        index->knnSearch(Matrix<ElementType>(testset, tcount, index->veclen()),
                         m_indices,
                         m_dists, nn, SearchParams(flann_params->checks) );

        return 0;
    }
    catch (std::runtime_error& e) {
        Logger::error("Caught exception: %s\n",e.what());
        return -1;
    }

    return -1;
}

template<typename T, typename R>
int _flann_find_nearest_neighbors_index(flann_index_t index_ptr, T* testset, int tcount,
                                        int* result, R* dists, int nn, FLANNParameters* flann_params)
{
    if (flann_distance_type==FLANN_DIST_EUCLIDEAN) {
        return __flann_find_nearest_neighbors_index<L2<T> >(index_ptr, testset, tcount, result, dists, nn, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_MANHATTAN) {
        return __flann_find_nearest_neighbors_index<L1<T> >(index_ptr, testset, tcount, result, dists, nn, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_MINKOWSKI) {
        return __flann_find_nearest_neighbors_index<MinkowskiDistance<T> >(index_ptr, testset, tcount, result, dists, nn, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_HIST_INTERSECT) {
        return __flann_find_nearest_neighbors_index<HistIntersectionDistance<T> >(index_ptr, testset, tcount, result, dists, nn, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_HELLINGER) {
        return __flann_find_nearest_neighbors_index<HellingerDistance<T> >(index_ptr, testset, tcount, result, dists, nn, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_CHI_SQUARE) {
        return __flann_find_nearest_neighbors_index<ChiSquareDistance<T> >(index_ptr, testset, tcount, result, dists, nn, flann_params);
    }
    else if (flann_distance_type==FLANN_DIST_KULLBACK_LEIBLER) {
        return __flann_find_nearest_neighbors_index<KL_Divergence<T> >(index_ptr, testset, tcount, result, dists, nn, flann_params);
    }
    else {
        Logger::error( "Distance type unsupported in the C bindings, use the C++ bindings instead\n");
        return -1;
    }
}

int flann_find_nearest_neighbors_index(flann_index_t index_ptr, float* testset, int tcount, int* result, float* dists, int nn, FLANNParameters* flann_params);

int flann_find_nearest_neighbors_index_float(flann_index_t index_ptr, float* testset, int tcount, int* result, float* dists, int nn, FLANNParameters* flann_params);

int flann_find_nearest_neighbors_index_double(flann_index_t index_ptr, double* testset, int tcount, int* result, double* dists, int nn, FLANNParameters* flann_params);

int flann_find_nearest_neighbors_index_byte(flann_index_t index_ptr, unsigned char* testset, int tcount, int* result, float* dists, int nn, FLANNParameters* flann_params);

int flann_find_nearest_neighbors_index_int(flann_index_t index_ptr, int* testset, int tcount, int* result, float* dists, int nn, FLANNParameters* flann_params);

#endif // _NI_LEGACY_FUNC_RECOGNITION_FLANN_H_
