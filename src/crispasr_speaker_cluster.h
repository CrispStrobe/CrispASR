// crispasr_speaker_cluster.h — agglomerative single-linkage clustering
// on L2-normalized speaker embeddings (issue #107 P3b).
//
// Given N speaker embeddings (each `dim` floats, expected L2-normalized
// since cosine similarity == dot product in that regime), agglomerate
// them into clusters until either (a) the maximum remaining pairwise
// similarity falls below `merge_threshold`, or (b) the cluster count
// drops to `max_speakers`.
//
// Returns a vector of length N with a cluster ID in [0, K) per input,
// where K is the final cluster count. The cluster IDs are assigned in
// first-appearance order so callers can use them as stable speaker
// indices.
//
// Threshold rule of thumb for TitaNet-Large: pairs above ~0.5 cosine
// are usually the same physical speaker; pairs below ~0.4 are usually
// different. We use 0.5 as the default merge threshold so a single
// embedder failure (a one-second noisy clip) doesn't collapse two
// real speakers.

#pragma once

#include <vector>

std::vector<int> crispasr_agglomerative_cluster(const std::vector<float>& embeddings, // N * dim floats, row-major
                                                int n, int dim, float merge_threshold = 0.5f, int max_speakers = 32);
