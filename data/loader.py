import json

def load_jsonl(path):
    data = []

    with open(path, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            data.append(obj)

    return data