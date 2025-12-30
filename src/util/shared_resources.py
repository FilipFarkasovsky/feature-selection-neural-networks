resources = None


class SharedResources:
    @staticmethod
    def set_resources(datasets, lock):
        global resources
        resources = {
        "datasets": datasets,
        "lock": lock
        }

    @staticmethod
    def get():
        global resources
        return resources
