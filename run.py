import argparse
import uvicorn


def run(port: int = 8008):
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


def create_parser():
    parser = argparse.ArgumentParser(description="Server running argument")
    parser.add_argument("--port", help="Set the port", type=int, default=8008)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    run(port=args.port)
