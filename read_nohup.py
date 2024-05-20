import argparse

def read_nohup(file_path: str, head: int = 10, tail: int = 10):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
        file_size = len(data) / 1024 / 1024
        print(f"nohup.out size: {file_size:.2f} MB")
        data = data.strip().split("\n")
        print("\n".join(data[:head]))
        print("\n".join(data[-tail:]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="nohup.out")
    parser.add_argument("--head", type=int, default=10)
    parser.add_argument("--tail", type=int, default=10)
    args = parser.parse_args()
    read_nohup(args.file, args.head, args.tail)