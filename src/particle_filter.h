#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"
#include<stdlib.h>
#include "map.h"
#include <list>

struct Particle {
	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};

struct single_landmark {
	double x_f;
	double y_f;
};

struct feature
{
	double* descr;
};

struct kd_node
{
	int ki;
	double kv;
	int leaf;
	struct feature* features;
	int n;
	int kd_left;
	int kd_right;
};

struct pq_node
{
	int ki;
	double kv;
	int pos;
	int key;
};

struct min_pq
{
	struct pq_node pq_array[100];
	int n;
};


class ParticleFilter {
	int num_particles;
	bool is_initialized;
	double x_min, x_mid, x_max, y_min, y_mid, y_max;
	std::list<single_landmark> map_1u, map_2u, map_3u, map_4u, map_12u, map_23u, map_34u, map_14u;
	std::vector<double> weights;
	std::list<single_landmark> p_obs;

public:
	std::vector<Particle> particles;

	ParticleFilter() : num_particles(0), is_initialized(false) {}

	~ParticleFilter() {
	}

	void init(double x, double y, double theta, double std[], const Map &map_landmarks);
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	void updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks);
	void resample();
	void kd_node_init(struct feature* featm, int n, int idx, struct kd_node (&kd)[100]);
	void expand_kd_node_subtree(int idx, struct kd_node (&kd)[100]);
	void assign_part_key(int idx, struct kd_node (&kd)[100]);
	double median_select(double*, int);
	double rank_select(double*, int, int);
	void insertion_sort(double*, int);
	int partition_array(double*, int, double);
	void partition_features(int idx, struct kd_node (&kd)[100]);
	int explore_to_leaf(int, struct single_landmark tar, struct min_pq &min_pq, struct kd_node(&kd)[100]);
	int insert_into_nbr_array(struct feature* feat_, int n, struct single_landmark tar, struct feature** _nbrs);
	void restore_minpq_order(struct min_pq &min_pq, int, int);
	void decrease_pq_node_key(struct min_pq &min_pq, int, int);
	void kdtree_build(struct feature* features, int n, struct kd_node (&kd)[100]);
	void kdtree_bbf_knn(struct single_landmark tar, struct kd_node(&kd)[100], struct feature** _nbrs, struct min_pq &min_pq);
	void minpq_insert(struct min_pq &min_pq, struct kd_node* data, int key, int pos);
	int minpq_extract_min(struct min_pq &min_pq);
	void SetAssociations(Particle& particle, const std::vector<int>& associations,
		const std::vector<double>& sense_x, const std::vector<double>& sense_y);
	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);
	const bool initialized() const {
		return is_initialized;
	}
};

#endif 