#include "ni/legacy/func_recognition_flann.h"

void flann_log_verbosity(int level)
{
    flann::log_verbosity(level);
}

void init_flann_parameters(FLANNParameters* p)
{
    if (p != NULL) {
        flann_log_verbosity(p->log_level);
        if (p->random_seed>0) {
            seed_random(p->random_seed);
        }
    }
}

flann::IndexParams create_parameters(FLANNParameters* p)
{
    flann::IndexParams params;

    params["algorithm"] = p->algorithm;

    params["checks"] = p->checks;
    params["cb_index"] = p->cb_index;
    params["eps"] = p->eps;

    if (p->algorithm == FLANN_INDEX_KDTREE) {
        params["trees"] = p->trees;
    }

    if (p->algorithm == FLANN_INDEX_KDTREE_SINGLE) {
        params["trees"] = p->trees;
        params["leaf_max_size"] = p->leaf_max_size;
    }

    if (p->algorithm == FLANN_INDEX_KDTREE_CUDA) {
        params["leaf_max_size"] = p->leaf_max_size;
    }

    if (p->algorithm == FLANN_INDEX_KMEANS) {
        params["branching"] = p->branching;
        params["iterations"] = p->iterations;
        params["centers_init"] = p->centers_init;
    }

    if (p->algorithm == FLANN_INDEX_AUTOTUNED) {
        params["target_precision"] = p->target_precision;
        params["build_weight"] = p->build_weight;
        params["memory_weight"] = p->memory_weight;
        params["sample_fraction"] = p->sample_fraction;
    }

    if (p->algorithm == FLANN_INDEX_HIERARCHICAL) {
        params["branching"] = p->branching;
        params["centers_init"] = p->centers_init;
        params["trees"] = p->trees;
        params["leaf_size"] = p->leaf_max_size;
    }

    if (p->algorithm == FLANN_INDEX_LSH) {
        params["table_number"] = p->table_number_;
        params["key_size"] = p->key_size_;
        params["multi_probe_level"] = p->multi_probe_level_;
    }

    params["log_level"] = p->log_level;
    params["random_seed"] = p->random_seed;

    return params;
}

flann_index_t flann_build_index(float* dataset, int rows, int cols, float* speedup, FLANNParameters* flann_params)
{
    return _flann_build_index<float>(dataset, rows, cols, speedup, flann_params);
}

int flann_find_nearest_neighbors_index(flann_index_t index_ptr, float* testset, int tcount, int* result, float* dists, int nn, FLANNParameters* flann_params)
{
    return _flann_find_nearest_neighbors_index(index_ptr, testset, tcount, result, dists, nn, flann_params);
}

int flann_find_nearest_neighbors_index_float(flann_index_t index_ptr, float* testset, int tcount, int* result, float* dists, int nn, FLANNParameters* flann_params)
{
    return _flann_find_nearest_neighbors_index(index_ptr, testset, tcount, result, dists, nn, flann_params);
}

int flann_find_nearest_neighbors_index_double(flann_index_t index_ptr, double* testset, int tcount, int* result, double* dists, int nn, FLANNParameters* flann_params)
{
    return _flann_find_nearest_neighbors_index(index_ptr, testset, tcount, result, dists, nn, flann_params);
}

int flann_find_nearest_neighbors_index_byte(flann_index_t index_ptr, unsigned char* testset, int tcount, int* result, float* dists, int nn, FLANNParameters* flann_params)
{
    return _flann_find_nearest_neighbors_index(index_ptr, testset, tcount, result, dists, nn, flann_params);
}

int flann_find_nearest_neighbors_index_int(flann_index_t index_ptr, int* testset, int tcount, int* result, float* dists, int nn, FLANNParameters* flann_params)
{
    return _flann_find_nearest_neighbors_index(index_ptr, testset, tcount, result, dists, nn, flann_params);
}
