from abc import ABC, abstractmethod


class BagReader(object):
    def __init__(self, bag_name):
        self._name = bag_name 

    @abstractmethod
    def read():
        pass

    @property
    def name(self):
        return self._name