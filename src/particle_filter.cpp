/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUMBER_OF_PARTICLES 100

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = NUMBER_OF_PARTICLES;
	weights.resize(num_particles);
	particles.resize(num_particles);


	std::random_device rd;

	std::mt19937 gen(rd());

	std::normal_distribution<> nd_x(0.0, std[0]);
	std::normal_distribution<> nd_y(0.0, std[1]);
	std::normal_distribution<> nd_t(0.0, std[2]);


	for (int i=0; i<particles.size(); i++)
	{
		weights[i] = 1.0;

		particles[i].id = i;

		particles[i].x = x + nd_x(gen);
		particles[i].y = y + nd_y(gen);
		particles[i].theta = theta + nd_t(gen);

		particles[i].weight = 1.0;
  	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i=0; i<particles.size(); i++)
	  {
	    double dist = velocity * delta_t;

	    double theta = particles[i].theta;

	    std::random_device rd;
	    std::mt19937 gen(rd());

	    std::normal_distribution<> nd_x(0.0, std_pos[0]);
	    std::normal_distribution<> nd_y(0.0, std_pos[1]);
	    std::normal_distribution<> nd_t(0.0, std_pos[2]);

	    // Going Straignt?
	    if (fabs(yaw_rate) < 0.001)
	    {
	      particles[i].x += dist * cos(theta) + nd_x(gen);
	      particles[i].y += dist * sin(theta) + nd_y(gen);
	    }
	    else
	    {
	      particles[i].x += velocity * (sin(theta + yaw_rate * delta_t) - sin(theta)) / yaw_rate + nd_x(gen);
	      particles[i].y += velocity * (cos(theta) - cos(theta + yaw_rate * delta_t)) / yaw_rate + nd_y(gen);
	      particles[i].theta += yaw_rate * delta_t + nd_t(gen);
	    }
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	std::vector<LandmarkObs> result;
	  for (int i=0; i<observations.size(); i++)
	  {
	    int map_index;
	    double dist_comp = 1000000;
	    for (int j=0; j<predicted.size(); j++)
	    {
	      double delta_x = predicted[j].x - observations[i].x;
	      double delta_y = predicted[j].y - observations[i].y;
	      double d = delta_x * delta_x + delta_y * delta_y; 
	      if (d < dist_comp)
	      {
		map_index = j;
		dist_comp = d;
	      } 
	    }
	    LandmarkObs l = {map_index, observations[i].x, observations[i].y};
	    result.push_back(l);
	  }
	observations = result;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	std::vector<Particle> new_particles;
	  std::vector<double> new_weights;
	  for (int i=0; i<particles.size(); i++)
	  {
	    double p_x = particles[i].x;
	    double p_y = particles[i].y;
	    double p_theta = particles[i].theta;

	    // Landmarks within the sensor range of the particle
	    std::vector<LandmarkObs> map_landmarks_in_range;
	    for (int j=0; j<map_landmarks.landmark_list.size(); j++)
	    {
	      int index = map_landmarks.landmark_list[j].id_i;
	      double m_x = map_landmarks.landmark_list[j].x_f;
	      double m_y = map_landmarks.landmark_list[j].y_f;
	      if (dist(p_x, p_y, m_x, m_y) < sensor_range)
	      {
		LandmarkObs l = {index, m_x, m_y};
		map_landmarks_in_range.push_back(l); 
	      }
	    }

	   // Observation coordinates to map coordinates
	    std::vector<LandmarkObs> observations_in_map_coordinates;
	    for (int j=0; j<observations.size(); j++)
	    {
	      int index = observations[j].id;
	      double o_x = observations[j].x;
	      double o_y = observations[j].y;
	      double m_x = p_x + o_x * cos(p_theta) - o_y * sin(p_theta);
	      double m_y = p_y + o_x * sin(p_theta) + o_y * cos(p_theta); 
	      LandmarkObs l = {index, m_x, m_y};
	      observations_in_map_coordinates.push_back(l); 
	    }
	 // Use the data association method 
	    dataAssociation(map_landmarks_in_range, observations_in_map_coordinates);

	    // Compute the new weight based on two vectors
	    double new_weight = 1.0; 
	    for (int j=0; j<observations_in_map_coordinates.size(); j++)
	    {
	      double o_x = observations_in_map_coordinates[j].x - p_x;
	      double o_y = observations_in_map_coordinates[j].y - p_y;
	      int map_index = observations_in_map_coordinates[j].id;
	      double m_x = map_landmarks_in_range[map_index].x - p_x;
	      double m_y = map_landmarks_in_range[map_index].y - p_y;
	      double o_length = sqrt(o_x * o_x + o_y * o_y);
	      double o_angle = atan2(o_y, o_x);
	      double m_length = sqrt(m_x * m_x + m_y * m_y);
	      double m_angle = atan2(m_y, m_x);
	      double delta_length = o_length - m_length;
	      double delta_angle = o_angle - m_angle; 	

	      // Bivariate Gaussian
	      double num_a = delta_length * delta_length / (2.0 * std_landmark[0] * std_landmark[0]);
	      double num_b = delta_angle * delta_angle / (2.0 * std_landmark[1] * std_landmark[1]);
	      double numerator = exp(-1.0 * (num_a + num_b));
	      double denominator = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
	      new_weight = numerator / denominator;
	    } 
	    particles[i].weight = new_weight;
	    weights[i] = new_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// Normalize weights
	  double total_weight = 0.0;
	  for (int i=0; i<particles.size(); i++)
	  {
	    total_weight += particles[i].weight;
	  }
	  for (int i=0; i<particles.size(); i++)
	  {
	    weights[i] /= total_weight;
	    particles[i].weight /= total_weight;
	  }

	  // Max of these weights
	  double max_weight = 0.0;
	  for (int i=0; i<particles.size(); i++)
	  {
	    if (weights[i] > max_weight)
	    {
	      max_weight = weights[i];
	    }
	}

	// Resample
	  double beta = 0.0;
	  int particles_size = particles.size();

	  std::random_device rd1;
	  std::mt19937 gen1(rd1());
	  std::uniform_int_distribution<> uniform_int(0, particles_size);
	  int index = uniform_int(gen1);

	  std::vector<Particle> new_particles;
	  std::vector<double> new_weights;

	  std::random_device rd2;
	  std::mt19937 gen2(rd2());
	  std::uniform_real_distribution<> uniform_double(0.0, 2.0 * max_weight);

	  for (int i=0; i<num_particles; i++)
	  {
	    beta += uniform_double(gen2);
	    while (weights[index] < beta)
	    {
	      beta -= weights[index];
	      index = (index + 1) % particles_size;
	    }
	    Particle p = {i, particles[index].x, particles[index].y, particles[index].theta, 1.0};
	    new_particles.push_back(p); 
	    new_weights.push_back(1.0);
	  }
	  particles = new_particles;
	weights = new_weights;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
