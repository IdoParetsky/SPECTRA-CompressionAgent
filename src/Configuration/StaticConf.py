from src.Configuration.ConfigurationValues import ConfigurationValues


class StaticConf:
    __instance = None
    conf_values: ConfigurationValues

    @staticmethod
    def get_instance() -> 'StaticConf':
        """ Static access method. """
        if StaticConf.__instance is None:
            StaticConf()
        return StaticConf.__instance

    def __init__(self, conf_values: ConfigurationValues):
        """ Virtually private constructor. """
        if StaticConf.__instance is not None:
            pass
            # raise Exception("This class is a singleton!")
        else:
            StaticConf.__instance = self
            self.conf_values = conf_values
