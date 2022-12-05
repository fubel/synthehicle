import carla


class WeatherSelector:
    def __init__(self):
        self.cloudiness = None  # 0.0 to 100.0
        self.precipitation = None  # 0.0 to 100.0
        self.precipitation_deposits = None  # 0.0 to 100.0
        self.wind_intensity = None  # 0.0 to 100.0
        self.sun_azimuth_angle = None  # 0.0 to 360.0
        self.sun_altitude_angle = None  # -90.0 to 90.0

    def get_weather_options(self):
        return [
            self.day(),
            self.dawn(),
            self.rainy(),
            self.night(),
        ]

    def day(self):
        self.cloudiness = 20.0
        self.precipitation = 0.0
        self.precipitation_deposits = 0.0
        self.wind_intensity = 2.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = 35.0
        self.fog_density = 2.0
        self.fog_distance = 0.75
        self.fog_falloff = 0.1
        self.wetness = 0.0
        self.scattering_intensity = 1.0
        self.mie_scattering_scale = 0.03
        self.rayleigh_scattering_scale = 0.0331

        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
            self.fog_density,
            self.fog_distance,
            self.fog_falloff,
            self.wetness,
            self.scattering_intensity,
            self.mie_scattering_scale,
            self.rayleigh_scattering_scale,
        ]

    def dawn(self):
        self.cloudiness = 20.0
        self.precipitation = 0.0
        self.precipitation_deposits = 0.0
        self.wind_intensity = 2.0
        self.sun_azimuth_angle = 200.0
        self.sun_altitude_angle = 5.0
        self.fog_density = 2.0
        self.fog_distance = 0.75
        self.fog_falloff = 0.1
        self.wetness = 0.0
        self.scattering_intensity = 1.0
        self.mie_scattering_scale = 0.03
        self.rayleigh_scattering_scale = 0.0331

        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
            self.fog_density,
            self.fog_distance,
            self.fog_falloff,
            self.wetness,
            self.scattering_intensity,
            self.mie_scattering_scale,
            self.rayleigh_scattering_scale,
        ]

    def rainy(self):
        self.cloudiness = 100.0
        self.precipitation = 95.0
        self.precipitation_deposits = 0.0
        self.wind_intensity = 40.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = 20.0
        self.fog_density = 20.0
        self.fog_distance = 0.9
        self.fog_falloff = 0.1
        self.wetness = 0.0
        self.scattering_intensity = 1.0
        self.mie_scattering_scale = 0.03
        self.rayleigh_scattering_scale = 0.0331

        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
            self.fog_density,
            self.fog_distance,
            self.fog_falloff,
            self.wetness,
            self.scattering_intensity,
            self.mie_scattering_scale,
            self.rayleigh_scattering_scale,
        ]

    def night(self):
        self.cloudiness = 0.0
        self.precipitation = 0.0
        self.precipitation_deposits = 0.0
        self.wind_intensity = 2.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = -45.0
        self.fog_density = 0.0
        self.fog_distance = 0.75
        self.fog_falloff = 0.1
        self.wetness = 0.0
        self.scattering_intensity = 1.0
        self.mie_scattering_scale = 0.03
        self.rayleigh_scattering_scale = 0.0331

        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
            self.fog_density,
            self.fog_distance,
            self.fog_falloff,
            self.wetness,
            self.scattering_intensity,
            self.mie_scattering_scale,
            self.rayleigh_scattering_scale,
        ]

    def midday(self):
        self.cloudiness = 30.0
        self.precipitation = 0.0
        self.precipitation_deposits = 60.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = 80  # 80.0  # 45
        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
        ]

    def afternoon(self):
        self.cloudiness = 50.0
        self.precipitation = 0.0
        self.precipitation_deposits = 40.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = -40.0
        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
        ]

    def default(self):
        self.cloudiness = -1.0
        self.precipitation = -1.0
        self.precipitation_deposits = -1.0
        self.wind_intensity = -1.0
        self.sun_azimuth_angle = -1.0
        self.sun_altitude_angle = -1.0
        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
        ]

    def almost_night(self):
        self.cloudiness = 30.0
        self.precipitation = 30.0
        self.precipitation_deposits = 0.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = 0.0
        self.sun_altitude_angle = -60.0
        self.wetness = 0.0
        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
        ]

    def customization(
        self,
        cloudiness=10.0,
        precipitation=10,
        wind_intensity=10,
        sun_azimuth_angle=10,
        sun_altitude_angle=10,
    ):

        return [
            cloudiness,
            precipitation,
            wind_intensity,
            sun_azimuth_angle,
            sun_altitude_angle,
        ]

    def HardRainNight(self):
        self.cloudiness = 100.0
        self.precipitation = 100.0
        self.precipitation_deposits = 90.0
        self.wind_intensity = 100.0
        self.sun_azimuth_angle = -1.0
        self.sun_altitude_angle = -90.0
        self.fog_density = 100.0
        self.fog_distance = 0.75
        self.fog_falloff = 0.1
        self.wetness = 100.0
        self.scattering_intensity = 1.00
        self.mie_scattering_scale = 0.03
        self.rayleigh_scattering_scale = 0.03

        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
            self.fog_density,
            self.fog_distance,
            self.fog_falloff,
            self.wetness,
            self.scattering_intensity,
            self.mie_scattering_scale,
            self.rayleigh_scattering_scale,
        ]

    def SoftRainNight(self):
        self.cloudiness = 60.0
        self.precipitation = 30.0
        self.precipitation_deposits = 50.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = -1.0
        self.sun_altitude_angle = -90.0
        self.fog_density = 60.0
        self.fog_distance = 0.75
        self.fog_falloff = 0.1
        self.wetness = 60.0
        self.scattering_intensity = 1.00
        self.mie_scattering_scale = 0.03
        self.rayleigh_scattering_scale = 0.03

        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
            self.fog_density,
            self.fog_distance,
            self.fog_falloff,
            self.wetness,
            self.scattering_intensity,
            self.mie_scattering_scale,
            self.rayleigh_scattering_scale,
        ]

    def HardRainNoon(self):
        self.cloudiness = 100.0
        self.precipitation = 100.0
        self.precipitation_deposits = 90.0
        self.wind_intensity = 100.0
        self.sun_azimuth_angle = -1.0
        self.sun_altitude_angle = 20.0
        self.fog_density = 7.0
        self.fog_distance = 0.75
        self.fog_falloff = 0.1
        self.wetness = 0.0
        self.scattering_intensity = 1.00
        self.mie_scattering_scale = 0.03
        self.rayleigh_scattering_scale = 0.03

        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
            self.fog_density,
            self.fog_distance,
            self.fog_falloff,
            self.wetness,
            self.scattering_intensity,
            self.mie_scattering_scale,
            self.rayleigh_scattering_scale,
        ]

    def SoftRainNoon(self):
        self.cloudiness = 20.0
        self.precipitation = 30.0
        self.precipitation_deposits = 50.0
        self.wind_intensity = 30.0
        self.sun_azimuth_angle = -1.0
        self.sun_altitude_angle = 45.0
        self.fog_density = 3.0
        self.fog_distance = 0.75
        self.fog_falloff = 0.1
        self.wetness = 0.0
        self.scattering_intensity = 1.00
        self.mie_scattering_scale = 0.03
        self.rayleigh_scattering_scale = 0.03

        return [
            self.cloudiness,
            self.precipitation,
            self.precipitation_deposits,
            self.wind_intensity,
            self.sun_azimuth_angle,
            self.sun_altitude_angle,
            self.fog_density,
            self.fog_distance,
            self.fog_falloff,
            self.wetness,
            self.scattering_intensity,
            self.mie_scattering_scale,
            self.rayleigh_scattering_scale,
        ]

    def set_weather(world, weather_option):
        weather = carla.WeatherParameters(*weather_option)
        world.set_weather(weather)
