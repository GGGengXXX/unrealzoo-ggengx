import numpy as np
from utils import getP

class randomPeopleController:
    def __init__(self):
        self.epochdelay = np.random.randint(1,5)
        self.iter = 0
        self.action = self.get_random_action()

    def update_delay(self):
        self.epochdelay = np.random.randint(1,5)

    def get_random_action(self):
        rol, forward = 0, 0
        headup = 0
        ani = 0
        if getP(0.1):
            rol = -30
        elif getP(0.13):
            rol = 30
        else:
            rol = 0

        if getP(0.8):
            forward = 100
        elif getP(0.2):
            forward = -100

        if getP(0.1):
            headup = 1
        elif getP(0.15):
            headup = 2
        else:
            headup = 0

        if getP(0.1):
            ani = np.random.randint(0, 4)

        return ([rol, forward], headup, ani)

    def try_update(self):
        self.iter += 1
        if self.iter == self.epochdelay:
            self.update_delay()
            self.iter = 0
            self.action = self.get_random_action()

    def get_action(self):
        self.try_update()
        return self.action

class randomAnimalController(randomPeopleController):
    def __init__(self):
        super().__init__()

    def get_random_action(cls):
        rol, forward = 0, 0
        if getP(0.1):
            rol = -30
        elif getP(0.13):
            rol = 30
        else:
            rol = 0

        if getP(0.8):
            forward = 100
        elif getP(0.2):
            forward = -100

        return ([rol, forward], None, None)