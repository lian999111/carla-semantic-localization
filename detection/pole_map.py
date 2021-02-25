"""Implementation of pole map generation"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

from detection.utils import Pole
from carlasim.utils import TrafficSignType

# import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)


def gen_pole_map(poles_xy, traffic_signs, pole_map_config):
    """
    Generate a pole map.

    The generated pole map is essentially a list of Pole objects. Given the x-y points of poles,
    this function clusters them (using DBSCAN) and take the means of clusters as pole landmarks
    in the map. The type of each pole may be assigned based on the proximity of the passed-in
    list of TrafficSigns.

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

    if __debug__:
        pole_landmark_x = []
        pole_landmark_y = []
        traffic_sign_x = []
        traffic_sign_y = []
        labeled_x = []
        labeled_y = []

    # Calculate the mean of each cluster as the location of the pole
    for label in range(n_clusters):
        mean_location = np.mean(poles_xy[:, labels == label], axis=1)
        # Add to pole_map
        pole_map.append(Pole(mean_location[0], mean_location[1]))

        if __debug__:
            pole_landmark_x.append(mean_location[0])
            pole_landmark_y.append(mean_location[1])

            # ax.plot(poles_xy[0, labels == label],
            #         poles_xy[1, labels == label], '.', ms=0.5)

    # Assign types to poles based on proximity
    classification_config = pole_map_config['classification']
    pole_x = [pole.x for pole in pole_map]
    pole_y = [pole.y for pole in pole_map]
    kd_poles = KDTree(np.asarray([pole_x, pole_y]).T)

    for traffic_sign in traffic_signs:
        nearest_idc = kd_poles.query_ball_point(
            [traffic_sign.x, traffic_sign.y], classification_config['max_dist'])

        if len(nearest_idc) == 0:
            continue

        elif len(nearest_idc) == 1:
            pole_map[nearest_idc[0]].type = traffic_sign.type

        elif len(nearest_idc) > 1:
            nearest_idx = None
            nearest_dist = classification_config['max_dist']

            for idx in nearest_idc:
                curr_dist = np.linalg.norm(np.array([traffic_sign.x - pole_map[idx].x,
                                                     traffic_sign.y - pole_map[idx].y]))
                if curr_dist < nearest_dist:
                    nearest_idx = idx

            pole_map[nearest_idx].type = traffic_sign.type

    if __debug__:
        plot_pole_map(pole_map, traffic_signs)

    return pole_map


def plot_pole_map(pole_map, traffic_signs):
    """Plot pole map along with traffic signs from Carla.

    Input:
        pole_map (list): List of Pole objects representing the map.
        traffic_signs: List of TrafficSigns.
    """
    traffic_sign_x = []
    traffic_sign_y = []
    unknown_type_x = []
    unknown_type_y = []
    labeled_x = []
    labeled_y = []

    for traffic_sign in traffic_signs:
        traffic_sign_x.append(traffic_sign.x)
        traffic_sign_y.append(traffic_sign.y)

    for pole in pole_map:
        if pole.type == TrafficSignType.Unknown:
            unknown_type_x.append(pole.x)
            unknown_type_y.append(pole.y)
        else:
            labeled_x.append(pole.x)
            labeled_y.append(pole.y)

    fig, ax = plt.subplots(1, 1)
    unknown_poles_plot = ax.plot(unknown_type_x, unknown_type_y,
                                 'g.', ms=6, label='pole landmark')[0]
    landmark_objs_plot = ax.plot(traffic_sign_x, traffic_sign_y,
                                 'bx', ms=6, label='landmark obj')[0]
    labeled_poles_plot = ax.plot(labeled_x, labeled_y,
                                 'r.', ms=6, label='labeled pole')[0]
    max_x = max(unknown_type_x)
    min_x = min(unknown_type_x)
    max_y = max(traffic_sign_y)
    min_y = min(traffic_sign_y)
    ax.set_xlim(min_x-10, max_x+10)
    ax.set_ylim(min_y-10, max_y+10)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.legend(framealpha=1.0)
    ax.tick_params(axis='y', which='major', pad=15)
    fig.savefig('pole_map_example.svg')
    plt.show()
