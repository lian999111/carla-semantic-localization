# Implementation of pole map generation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

from detection.utils import Pole

def gen_pole_map(poles_xy, traffic_signs, pole_map_config):
    """
    Generate a pole map.

    The generated pole map is essentially a list of Pole objects. Given the x-y points of poles, 
    this function clusters them (using DBSCAN) and take the means of clusters as pole landmarks 
    in the map. The type of each pole may be assigned based on the proximity of the passed-in list of TrafficSigns.

    Input:
        poles_xy: 2-by-N numpy.array representating x-y coordinates of poles.
        traffic_signs: List of TrafficSigns.
        pole_map_config: Dict of pole map generation configuration.
    Output:
        List of Pole objects.
    """
    pole_map = []

    # Use DBSCAN to cluster pole points
    clustering_config = pole_map_config['clustering']
    pole_clustering = DBSCAN(eps=clustering_config['eps'],
                            min_samples=clustering_config['min_samples']).fit(poles_xy.T)

    labels = pole_clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    for label in range(n_clusters):
        mean_location = np.mean(poles_xy[:, labels == label], axis=1)
        # Add to pole_map
        pole_map.append(Pole(mean_location[0], mean_location[1]))

        if __debug__:
            plt.plot(mean_location[0], mean_location[1], 's')
            plt.plot(poles_xy[0, labels == label],
                    poles_xy[1, labels == label], '.', ms=0.5)

    # Assign types to poles based on proximity
    classification_config = pole_map_config['classification']
    pole_x = [pole.x for pole in pole_map]
    pole_y = [pole.y for pole in pole_map]
    kd_poles = KDTree(np.asarray([pole_x, pole_y]).T)

    for traffic_sign in traffic_signs:
        if __debug__:
            # Plot all traffic signs
            plt.plot(traffic_sign.x, traffic_sign.y, 'bs', ms=2)

        nearest_idc = kd_poles.query_ball_point([traffic_sign.x, traffic_sign.y], classification_config['perimeter'])
        if len(nearest_idc) == 1:
            pole_map[nearest_idc[0]].type = traffic_sign.type
            
            if __debug__:
                plt.plot(pole_map[nearest_idc[0]].x, pole_map[nearest_idc[0]].y, 'r+', ms=3)
        else:
            continue

    if __debug__:
        plt.legend()
        plt.show()
        
    return pole_map
