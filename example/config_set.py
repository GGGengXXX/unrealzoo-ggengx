class config:
    def __init__(self):
        pass

    def get_high_fps_1920x1080_config(self):
        return 50, [1920, 1280], 60.0

    def get_normal_config(self):
        return 1, [240,240], 5