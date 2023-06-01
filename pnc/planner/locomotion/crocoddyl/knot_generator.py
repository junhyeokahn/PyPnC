import numpy as np


def linear_knots(ini_pos, end_pos, number_of_waypoints):
    """
    Creates list of number_of_waypoints linear and  equidistant knots between
    the given initial and end positions.

    :param ini_pos: Initial position in world coordinates
    :param end_pos: Final position in world coordinates
    :param number_of_waypoints: Number of desired waypoints from ini_pos to end_pos
    """
    # end_pos += ini_pos
    equidistant_segment = (end_pos - ini_pos) / (number_of_waypoints-1)

    knots = []
    knots += [ini_pos]
    for i in np.arange(number_of_waypoints-1):
        knots += [ini_pos + (i+1) * equidistant_segment]
        # knots += [end_pos]
    return knots


def swing_knots(ini_pos, end_pos, number_of_waypoints, swing_height):
    """
    Creates list of number_of_waypoints equidistant knots between the given
    initial and end positions for a swing leg. This assumes the overall
    middle knot is going to be at the given swing height.
    """
    # equidistant_segment = (ini_pos + end_pos) / (number_of_waypoints-1)
    equidistant_segment = (end_pos - ini_pos) / (number_of_waypoints-1)
    half_waypoints = np.floor(number_of_waypoints/2)

    equidistant_up = equidistant_segment.copy()
    equidistant_up[2] += swing_height / half_waypoints
    equidistant_down = equidistant_segment.copy()
    equidistant_down[2] -= swing_height / half_waypoints
    knots = []
    knots += [ini_pos]
    for i in np.arange(number_of_waypoints-1):
        current_knot = knots[i]
        # adjust the height of swing foot waypoint
        if i < (number_of_waypoints-1)/2:
            equidistant_segment = equidistant_up
        else:
            equidistant_segment = equidistant_down
        knots += [current_knot + equidistant_segment]
    return knots


def linear_connection(ini_points, end_points, number_of_waypoints):
    waypoints = []
    for point_ini, point_end, n_waypoints in zip(ini_points, end_points, number_of_waypoints):
        waypoints += linear_knots(point_ini, point_end, n_waypoints)
    return waypoints
