from lib.runner import Runner
from lib.args import Arguments

if __name__ == "__main__":
    args = Arguments().parse_args()
    args.sanitize()
    runner = Runner(args)
    runner.start()
