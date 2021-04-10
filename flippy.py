from lib.runner import Runner
from lib.args import ArgumentParser

if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    args.sanitize()
    runner = Runner(args)
