from argparse import ArgumentParser


class Runner:
    def __init__(self, args: ArgumentParser):
        """
        Precondition: assuming args.sanitize() was called.
        """
        self.args = args

    def start(self):

        pass
