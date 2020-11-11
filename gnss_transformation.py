import pymap3d as pm

# obtain CARLA map
map_carla = world.get_map()
map_carla_origin_geo = map_carla.transform_to_geolocation(carla.Location(0.0, 0.0, 0.0))

# Internally Carla performs geodetic (latitude, longitude, altitude) to Cartesian transformations which are not
# quite clear. The Carla documentation reveals that a Mercator projection might be used internally
# (see https://carla.org/Doxygen/html/df/ddb/GeoLocation_8cpp_source.html) however using this transformation did not
# yield the desired results. Therefore a geodetic to east-north-up (ENU) transformation is used
# (mentioned in https://github.com/carla-simulator/carla/issues/2737, implemented in helper function geo_carla2xyz_carla)
# for transforming the measurements of the CARLA GnssSensor to the local coordinate frame of Carla.
# The errors with respect to the ground truth (get_transform().location) are in the magnitude of millimeters which is
# quite high and cannot be explained by rounding errors.
#
# TODO: further investigations might be needed here. New release might solve this issue (see https://github.com/carla-simulator/carla/issues/1848)



def geo_carla2xyz_carla(lat, lon, alt):
    # transforms geodetic location from carla.GnssMeasurement to location in cartesian Carla map frame. However,
    # transformed location and ground truth deviate from each other (see above).

    # after this transformation of the latitude, GNSS projection and ego_vehicle.get_transform().location are "closer"
    lat = pm.geocentric2geodetic(lat, alt, pm.Ellipsoid('wgs84'), deg=True)

    x_enu, y_enu, z_enu = pm.geodetic2enu(lat, lon, alt, map_carla_origin_geo.latitude, map_carla_origin_geo.longitude, map_carla_origin_geo.altitude, pm.Ellipsoid('wgs84'), deg=True)

    # y-coordinate in Carla coordinate system is flipped (see https://github.com/carla-simulator/carla/issues/2737)
    return x_enu, -y_enu, z_enu