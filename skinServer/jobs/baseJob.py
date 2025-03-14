
errorInfo = {
    1: "default error.",
    101: "BSA hand reject.",
    102: "No hand base find.",
}


class BaseJob(object):
    def __init__(self):
        """ init your model here. """
        pass

    def run(self, request):
        """ Implement it for it`s function. """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.run(*args)
