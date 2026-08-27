"""Tiny calculator CLI."""
import argparse
import sys

from ops import add, sub, mul, div


def main():
    parser = argparse.ArgumentParser(description="calculator")
    parser.add_argument("a", type=float)
    parser.add_argument("b", type=float)
    parser.add_argument("--add", action="store_true")
    parser.add_argument("--sub", action="store_true")
    parser.add_argument("--mul", action="store_true")
    parser.add_argument("--div", action="store_true")
    args = parser.parse_args()

    if args.add:
        result = add(args.a, args.b)
    elif args.sub:
        result = sub(args.a, args.b)
    elif args.mul:
        result = mul(args.a, args.b)
    elif args.div:
        result = div(args.a, args.b)
    else:
        print("error: pick an operation flag", file=sys.stderr)
        sys.exit(2)

    print(result)


if __name__ == "__main__":
    main()
