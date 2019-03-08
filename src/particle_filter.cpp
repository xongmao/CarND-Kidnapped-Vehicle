#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <stdio.h>
#include <limits.h>
#include<stdlib.h>
#include <string.h>
#include <time.h>
#include <list>
#include "particle_filter.h"
using namespace std;

__inline int parent(int i)
{
	return (i - 1) / 2;
}

__inline int right(int i)
{
	return 2 * i + 2;
}

__inline int left(int i)
{
	return 2 * i + 1;
}

void ParticleFilter::minpq_insert(struct min_pq &min_pq, struct kd_node* data, int key, int pos)
{
	int n = min_pq.n;
	min_pq.pq_array[n].ki = data->ki;
	min_pq.pq_array[n].kv = data->kv;
	min_pq.pq_array[n].pos = pos;
	min_pq.pq_array[n].key = INT_MAX;
	decrease_pq_node_key(min_pq, min_pq.n, key);
	min_pq.n++;
}

int ParticleFilter::minpq_extract_min(struct min_pq &min_pq)
{
	int data;
	if (min_pq.n < 1) {
		return -1;
	}
	data = min_pq.pq_array[0].pos;
	min_pq.n--;
	min_pq.pq_array[0] = min_pq.pq_array[min_pq.n];
	restore_minpq_order(min_pq, 0, min_pq.n);
	return data;
}

void ParticleFilter::decrease_pq_node_key(struct min_pq &min_pq, int i, int key)
{
	struct pq_node tmp_pq;
	if (key > min_pq.pq_array[i].key) {
		return;
	}
	min_pq.pq_array[i].key = key;
	while (i > 0 && min_pq.pq_array[i].key < min_pq.pq_array[parent(i)].key) {
		tmp_pq = min_pq.pq_array[parent(i)];
		min_pq.pq_array[parent(i)] = min_pq.pq_array[i];
		min_pq.pq_array[i] = tmp_pq;
		i = parent(i);
	}
}

void ParticleFilter::restore_minpq_order(struct min_pq &min_pq, int i, int n)
{
	struct pq_node tmp_pq;
	int l, r, min = i;

	l = left(i);
	r = right(i);
	if (l < n) {
		if (min_pq.pq_array[l].key < min_pq.pq_array[i].key) {
			min = l;
		}
	}
	if (r < n) {
		if (min_pq.pq_array[r].key < min_pq.pq_array[min].key) {
			min = r;
		}
	}
	if (min != i) {
		tmp_pq = min_pq.pq_array[min];
		min_pq.pq_array[min] = min_pq.pq_array[i];
		min_pq.pq_array[i] = tmp_pq;
		restore_minpq_order(min_pq, min, n);
	}
}

void ParticleFilter::kdtree_build(struct feature* features, int n, struct kd_node (&kd)[100])
{

	kd_node_init(features, n, 0, kd);
	expand_kd_node_subtree(0, kd);
}

void ParticleFilter::kdtree_bbf_knn(struct single_landmark tar, struct kd_node(&kd)[100], struct feature** _nbrs, struct min_pq &min_pq)
{
	int expl;
	struct feature* tree_feat;
	int i, t = 0, n = 0;
	minpq_insert(min_pq, kd, 0, 0);
	while (min_pq.n > 0 && t < 200)
	{
		expl = minpq_extract_min(min_pq);
		expl = explore_to_leaf(expl, tar, min_pq, kd);
		for (i = 0; i < kd[expl].n; i++)
		{
			tree_feat = &kd[expl].features[i];
			n += insert_into_nbr_array(tree_feat, n, tar, _nbrs);
		}
		t++;
	}
}

void ParticleFilter::kd_node_init(struct feature* featm, int n, int idx, struct kd_node (&kd)[100])
{
	kd[idx].ki = -1;
	kd[idx].features = featm;
	kd[idx].n = n;
	kd[idx].kd_left = left(idx);
	kd[idx].kd_right = right(idx);
}

void ParticleFilter::expand_kd_node_subtree(int idx, struct kd_node (&kd)[100])
{
	if (kd[idx].n == 1 || kd[idx].n == 0)
	{
		kd[idx].leaf = 1;
		return;
	}
	assign_part_key(idx, kd);
	partition_features(idx, kd);

	if (kd[left(idx)].features)
		expand_kd_node_subtree(left(idx), kd);
	if (kd[right(idx)].features)
		expand_kd_node_subtree(right(idx), kd);
}

void ParticleFilter::assign_part_key(int idx, struct kd_node (&kd)[100])
{
	double kv, x, mean, var, var_max = 0;
	int n, i, j, ki = 0;
	double* tmp;
	n = kd[idx].n;

	for (j = 0; j < 2; j++)
	{
		mean = var = 0;
		for (i = 0; i < n; i++) {
			mean += kd[idx].features[i].descr[j];
		}
		mean /= n;
		for (i = 0; i < n; i++)
		{
			x = kd[idx].features[i].descr[j] - mean;
			var += x * x;
		}
		var /= n;

		if (var > var_max)
		{
			ki = j;
			var_max = var;
		}
	}
	tmp = (double*)(calloc(n, sizeof(double)));
	for (i = 0; i < n; i++) {
		tmp[i] = kd[idx].features[i].descr[ki];
	}
	kv = median_select(tmp, n);
	free(tmp);
	kd[idx].ki = ki;
	kd[idx].kv = kv;
}

double ParticleFilter::median_select(double* array, int n)
{
	return rank_select(array, n, (n - 1) / 2);
}

double ParticleFilter::rank_select(double* array, int n, int r)
{
	double* tmp_a, med;
	int gr_5, gr_tot, rem_elts, i, j;

	if (n == 1) {
		return array[0];
	}

	gr_5 = n / 5;
	gr_tot = (int)ceil(n / 5.0);
	rem_elts = n % 5;
	tmp_a = array;
	for (i = 0; i < gr_5; i++)
	{
		insertion_sort(tmp_a, 5);
		tmp_a += 5;
	}
	insertion_sort(tmp_a, rem_elts);

	tmp_a = (double*)(calloc(gr_tot, sizeof(double)));
	for (i = 0, j = 2; i < gr_5; i++, j += 5) {
		tmp_a[i] = array[j];
	}
	if (rem_elts) {
		tmp_a[i++] = array[n - 1 - rem_elts / 2];
	}
	med = rank_select(tmp_a, i, (i - 1) / 2);
	free(tmp_a);

	j = partition_array(array, n, med);
	if (r == j) {
		return med;
	}
	else if (r < j) {
		return rank_select(array, j, r);
	}
	else {
		array += j + 1;
		return rank_select(array, (n - j - 1), (r - j - 1));
	}
}


void ParticleFilter::insertion_sort(double* array, int n)
{
	double k;
	int i, j;
	for (i = 1; i < n; i++) {
		k = array[i];
		j = i - 1;
		while (j >= 0 && array[j] > k) {
			array[j + 1] = array[j];
			j -= 1;
		}
		array[j + 1] = k;
	}
}


int ParticleFilter::partition_array(double* array, int n, double pivot)
{
	double tmp_n;
	int p, i, j;
	i = -1;
	for (j = 0; j < n; j++) {
		if (array[j] <= pivot) {
			tmp_n = array[++i];
			array[i] = array[j];
			array[j] = tmp_n;
			if (array[i] == pivot) {
				p = i;
			}
		}
	}
	array[p] = array[i];
	array[i] = pivot;
	return i;
}

void ParticleFilter::partition_features(int idx, struct kd_node (&kd)[100])
{
	struct feature* features, tmp_f;
	double kv;
	int n, ki, p, i, j = -1;

	features = kd[idx].features;
	n = kd[idx].n;
	ki = kd[idx].ki;
	kv = kd[idx].kv;
	for (i = 0; i < n; i++) {
		if (features[i].descr[ki] <= kv) {
			tmp_f = features[++j];
			features[j] = features[i];
			features[i] = tmp_f;
			if (features[j].descr[ki] == kv) {
				p = j;
			}
		}
	}
	tmp_f = features[p];
	features[p] = features[j];
	features[j] = tmp_f;

	if (j == n - 1) {
		kd[idx].leaf = 1;
		return;
	}

	kd_node_init(features, j + 1, left(idx), kd);
	kd_node_init(features + (j + 1), (n - j - 1), right(idx), kd);
}

int ParticleFilter::explore_to_leaf(int kd_nodem, struct single_landmark tar, struct min_pq &min_pq, struct kd_node(&kd)[100])
{
	int unexpl, expl = kd_nodem;
	double kv;
	int ki;

	while (&kd[expl] && !kd[expl].leaf) {
		ki = kd[expl].ki;
		kv = kd[expl].kv;
		double go = ki == 0 ? tar.x_f : tar.y_f;
		if (go <= kv) {
			unexpl = right(expl);
			expl = left(expl);
		}
		else {
			unexpl = left(expl);
			expl = right(expl);
		}
		minpq_insert(min_pq, &kd[unexpl], abs(kv - go), unexpl);
	}

	return expl;
}

double dist(struct feature* feat_, struct single_landmark tar)
{
	double diff, res = 0.0;
	for (int i = 0; i<2; i++) {
		double go = i == 0 ? tar.x_f : tar.y_f;
		diff = feat_->descr[i] - go;
		res += diff * diff;
	}
	return res;
}

int ParticleFilter::insert_into_nbr_array(struct feature* feat_, int n, struct single_landmark tar, struct feature** _nbrs)
{
	double dn, df;
	if (n == 0) {
		_nbrs[0] = feat_;
		return 1;
	}
	df = dist(feat_, tar);
	dn = dist(_nbrs[0], tar);
	if (df >= dn) {
		return 1;
	}
	else {
		_nbrs[0] = feat_;
		return 1;
	}
}

void ParticleFilter::init(double x, double y, double theta, double std[], const Map &map_landmarks) {
	is_initialized = true;
	num_particles = 500;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	weights.resize(num_particles);
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		weights[i] = p.weight;
		particles.push_back(p);
	}

	int n = map_landmarks.landmark_list.size();
	x_min = map_landmarks.landmark_list[0].x_f;
    x_max = map_landmarks.landmark_list[0].x_f;
    y_min = map_landmarks.landmark_list[0].y_f;
    y_max = map_landmarks.landmark_list[0].y_f;
    for(int i=1; i<n; i++){
		if(x_max<map_landmarks.landmark_list[i].x_f){
			x_max = map_landmarks.landmark_list[i].x_f;
		}
		if(x_min>map_landmarks.landmark_list[i].x_f){
			x_min = map_landmarks.landmark_list[i].x_f;
		}
		if(y_max<map_landmarks.landmark_list[i].y_f){
			y_max = map_landmarks.landmark_list[i].y_f;
		}
		if(y_min>map_landmarks.landmark_list[i].y_f){
			y_min = map_landmarks.landmark_list[i].y_f;
		}
    }
	x_mid = (x_min + x_max) / 2.0;
	y_mid = (y_min + y_max) / 2.0;
	for (int i = 0; i < n; ++i) {
		double a = map_landmarks.landmark_list[i].x_f;
		double b = map_landmarks.landmark_list[i].y_f;
		if (a >= x_min && a<x_mid) {
			single_landmark ss;
			ss.x_f = a;
			ss.y_f = b;
			map_12u.push_back(ss);
		}
		if (b >= y_mid && b <= y_max) {
			single_landmark ss;
			ss.x_f = a;
			ss.y_f = b;
			map_23u.push_back(ss);
		}
		if (a >= x_mid && a <= x_max) {
			single_landmark ss;
			ss.x_f = a;
			ss.y_f = b;
			map_34u.push_back(ss);
		}
		if (b >= y_min && b<y_mid) {
			single_landmark ss;
			ss.x_f = a;
			ss.y_f = b;
			map_14u.push_back(ss);
		}
		if ((a >= x_min && a<x_mid) && (b >= y_min && b<y_mid)) {
			single_landmark ss;
			ss.x_f = a;
			ss.y_f = b;
			map_1u.push_back(ss);
		}
		if ((a >= x_min && a<x_mid) && (b >= y_mid && b <= y_max)) {
			single_landmark ss;
			ss.x_f = a;
			ss.y_f = b;
			map_2u.push_back(ss);
		}
		if ((a >= x_mid && a <= x_max) && (b >= y_mid && b <= y_max)) {
			single_landmark ss;
			ss.x_f = a;
			ss.y_f = b;
			map_3u.push_back(ss);
		}
		if ((a >= x_mid && a <= x_max) && (b >= y_min && b<y_mid)) {
			single_landmark ss;
			ss.x_f = a;
			ss.y_f = b;
			map_4u.push_back(ss);
		}
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double eps = 0.0001;
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	if (fabs(yaw_rate) <= eps) {
		for (int i = 0; i < num_particles; ++i) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);

		}
	}
	else {
		for (int i = 0; i < num_particles; ++i) {
			particles[i].x += velocity * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) / yaw_rate + dist_x(gen);
			particles[i].y += velocity * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) / yaw_rate + dist_y(gen);
			particles[i].theta += yaw_rate * delta_t + dist_theta(gen);

		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	int n = observations.size();
	int m = map_landmarks.landmark_list.size();
	double px_min, px_max, py_min, py_max, theta;
	double dif, mx, my, w, exponent;
	double gauss_norm = (1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]));
	int index;
	for (int k = 0; k < num_particles; ++k) {
		px_min = particles[k].x - sensor_range;
		px_max = particles[k].x + sensor_range;
		py_min = particles[k].y - sensor_range;
		py_max = particles[k].y + sensor_range;
		theta = particles[k].theta;
		index = 0;
		for (int i = 0; i < n; ++i) {
			mx = particles[k].x + cos(theta) * observations[i].x - sin(theta) * observations[i].y;
			my = particles[k].y + sin(theta) * observations[i].x + cos(theta) * observations[i].y;
			dif = dist(particles[k].x, particles[k].y, mx, my);
			if (dif > sensor_range) {
				continue;
			}
			single_landmark po;
			po.x_f = mx;
			po.y_f = my;
			p_obs.push_back(po);
			index++;
		}
		if(index==0){
			continue;
		}

		struct feature* feat = new struct feature[100];
		for (int i = 0; i < 100; i++) {
			feat[i].descr = (double *)(calloc(2, sizeof(double)));
		}
		w = 1.0;
		if (px_max<x_mid) {
			if (py_max<y_mid) {
				index = 0;
				for (list<single_landmark>::iterator i = map_1u.begin(); i != map_1u.end(); ++i) {
					dif = dist(particles[k].x, particles[k].y, i->x_f, i->y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = i->x_f;
					feat[index++].descr[1] = i->y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
			else if (py_min >= y_mid) {
				index = 0;
				for (list<single_landmark>::iterator i = map_2u.begin(); i != map_2u.end(); ++i) {
					dif = dist(particles[k].x, particles[k].y, i->x_f, i->y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = i->x_f;
					feat[index++].descr[1] = i->y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
			else {
				index = 0;
				for (list<single_landmark>::iterator i = map_12u.begin(); i != map_12u.end(); ++i) {
					dif = dist(particles[k].x, particles[k].y, i->x_f, i->y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = i->x_f;
					feat[index++].descr[1] = i->y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
		}
		else if (px_min >= x_mid) {
			if (py_max<y_mid) {
				index = 0;
				for (list<single_landmark>::iterator i = map_4u.begin(); i != map_4u.end(); ++i) {
					dif = dist(particles[k].x, particles[k].y, i->x_f, i->y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = i->x_f;
					feat[index++].descr[1] = i->y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
			else if (py_min >= y_mid) {
				index = 0;
				for (list<single_landmark>::iterator i = map_3u.begin(); i != map_3u.end(); ++i) {
					dif = dist(particles[k].x, particles[k].y, i->x_f, i->y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = i->x_f;
					feat[index++].descr[1] = i->y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
			else {
				index = 0;
				for (list<single_landmark>::iterator i = map_34u.begin(); i != map_34u.end(); ++i) {
					dif = dist(particles[k].x, particles[k].y, i->x_f, i->y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = i->x_f;
					feat[index++].descr[1] = i->y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
		}
		else {
			if (py_max<y_mid) {
				index = 0;
				for (list<single_landmark>::iterator i = map_14u.begin(); i != map_14u.end(); ++i) {
					dif = dist(particles[k].x, particles[k].y, i->x_f, i->y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = i->x_f;
					feat[index++].descr[1] = i->y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
			else if (py_min >= y_mid) {
				index = 0;
				for (list<single_landmark>::iterator i = map_23u.begin(); i != map_23u.end(); ++i) {
					dif = dist(particles[k].x, particles[k].y, i->x_f, i->y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = i->x_f;
					feat[index++].descr[1] = i->y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
			else {
				index = 0;
				for (int i = 0; i < m; ++i) {
					dif = dist(particles[k].x, particles[k].y, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f);
					if (dif > sensor_range) {
						continue;
					}
					feat[index].descr[0] = map_landmarks.landmark_list[i].x_f;
					feat[index++].descr[1] = map_landmarks.landmark_list[i].y_f;
				}
				if (index == 0) {
					p_obs.clear();
					continue;
				}
				struct kd_node kd[100];
				for (int i = 0; i < 100; i++) {
					kd[i].ki = -1;
					kd[i].kv = 0.0;
					kd[i].n = 0;
					kd[i].leaf = 0;
					kd[i].features = (struct feature*)malloc(sizeof(struct feature));
					memset(kd[i].features, 0, sizeof(struct feature));
					kd[i].kd_left = -1;
					kd[i].kd_right = -1;
				}
				kdtree_build(feat, index, kd);
				
				struct min_pq min_pq;
				min_pq.n = 0;
				struct pq_node pq_node;
				pq_node.ki = -1;
				pq_node.kv = 0.0;
				pq_node.pos = -1;
				pq_node.key = INT_MAX;
				for (int i = 0; i < 100; ++i) {
					min_pq.pq_array[i] = pq_node;
				}
				struct feature* _nbrs = (struct feature*)malloc(sizeof(struct feature));
				memset(_nbrs, 0, sizeof(struct feature));
				struct feature* p_nbrs;
				p_nbrs = _nbrs;
				
				for (list<single_landmark>::iterator i = p_obs.begin(); i != p_obs.end(); ++i) {
					struct single_landmark ss;
					ss.x_f = i->x_f;
					ss.y_f = i->y_f;
					kdtree_bbf_knn(ss, kd, &_nbrs, min_pq);
					exponent = pow(ss.x_f - _nbrs->descr[0], 2.0) / (2.0 * pow(std_landmark[0], 2.0)) + pow(ss.y_f - _nbrs->descr[1], 2.0) / (2.0 * pow(std_landmark[1], 2.0));
					w *= gauss_norm * exp(-exponent);
				}
				particles[k].weight = w;
				weights[k] = particles[k].weight;
				p_obs.clear();
				_nbrs = p_nbrs;
				free(_nbrs);
				_nbrs = NULL;
				for (int i = 0; i < 100; i++) {
					free(feat[i].descr);
					feat[i].descr = NULL;
				}
				delete[] feat;
				feat = NULL;
			}
		}
	}
}

void ParticleFilter::resample() {
	vector<Particle> reParticles;
	srand((unsigned)time(NULL));
	int index = rand() % num_particles;
	double beta = 0.0;
	double mw = particles[0].weight;
	for (int i = 1; i<num_particles; i++) {
		if (mw<particles[i].weight) {
			mw = particles[i].weight;
		}
	}
	for (int i = 0; i<num_particles; i++) {
		beta += (double)rand() / (double)(RAND_MAX / (2.0 * mw));
		while (beta > particles[index].weight) {
			beta -= particles[index].weight;
			index = (index + 1) % num_particles;
		}
		reParticles.push_back(particles[index]);
	}
	for (int i = 0; i<num_particles; i++) {
		particles[i] = reParticles[i];
		weights[i] = particles[i].weight;
	}
	reParticles.clear();
	
	//-------------------------
	std::random_device seed;
	std::mt19937 random_generator(seed());
	// sample particles based on their weight
	std::discrete_distribution<> sample(weights.begin(), weights.end());

	std::vector<Particle> new_particles(num_particles);
	for(auto & p : new_particles)
		p = particles[sample(random_generator)];
	particles = std::move(new_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
	const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);
	return s;
}