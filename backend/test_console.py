import requests
import time

API = "http://127.0.0.1:8000"


def main():
    cfg = requests.get(f"{API}/config").json()
    datasets = cfg["datasets"]
    models = cfg["models"]

    while True:
        print("\nDATASETS:")
        for i, d in enumerate(datasets):
            print(f"[{i+1}] {d}")
        ds = datasets[int(input("> ")) - 1]

        print("\nMODELS:")
        for i, m in enumerate(models):
            print(f"[{i+1}] {m}")
        mt = models[int(input("> ")) - 1]

        seq = input("\nSequence: ").strip()
        if not seq:
            seq = "ACGU" * 30

        payload = {
            "dataset": ds,
            "model_type": mt,
            "input_text": seq
        }

        t0 = time.time()
        res = requests.post(f"{API}/predict", json=payload).json()
        dt = (time.time() - t0) * 1000

        print(f"\nPrediction: {res['prediction']:.4f} [{res['label']}] ({dt:.1f} ms)")
        print(f"Heatmap length: {len(res['heatmap'])}")
        print(f"Polynomial arcs: {len(res['polynomial_arcs'])}")
        print(f"Spherical vectors: {len(res['sphere_vectors'])}")


if __name__ == "__main__":
    main()
