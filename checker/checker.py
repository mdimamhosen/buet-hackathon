import requests
import json
import sys
import argparse

BASE_URL = "http://localhost:8080"


def normalize(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        try:
            return round(float(value), 2)
        except:
            return value
    return value


def check_health():
    print("🔍 Checking /health ...")
    r = requests.get(f"{BASE_URL}/health", timeout=5)

    if r.status_code != 200:
        raise RuntimeError(f"Health check failed with status {r.status_code}")

    data = r.json()
    assert data.get("status") == "ok", "Health status is not ok"
    assert data.get("database") == "connected", "Database not connected"

    print("✅ Health check passed\n")


def run_tests():
    with open("test_cases.json") as f:
        tests = json.load(f)

    passed = 0
    total = len(tests)

    for idx, t in enumerate(tests, start=1):
        print(f"\n🧪 Test {idx}/{total}")
        print(f"❓ Question: {t['question']}")

        try:
            r = requests.post(
                f"{BASE_URL}/query",
                json={
                    "question": t["question"],
                    "llm": "gemini-2.5-flash"
                },
                timeout=30
            )
        except Exception as e:
            print("❌ Request failed:", e)
            continue

        if r.status_code != 200:
            print(f"❌ HTTP {r.status_code}")
            print(r.text)
            continue

        res = r.json()

        # Check result_type
        if res.get("result_type") != t["result_type"]:
            print("❌ result_type mismatch")
            print("Expected:", t["result_type"])
            print("Got     :", res.get("result_type"))
            continue

        exp_rows = t["expected"]["rows"]
        got_rows = res.get("rows")

        exp_norm = [[normalize(x) for x in row] for row in exp_rows]
        got_norm = [[normalize(x) for x in row] for row in got_rows]

        if exp_norm != got_norm:
            print("❌ Rows mismatch")
            print("Expected:")
            for row in exp_norm:
                print(" ", row)
            print("Got:")
            for row in got_norm:
                print(" ", row)
            continue

        print("✅ Test passed")
        passed += 1

    print("\n" + "=" * 40)
    print(f"✅ Passed {passed}/{total} tests")
    print("=" * 40)


def main():
    global BASE_URL

    parser = argparse.ArgumentParser(description="Run checker against a server")
    parser.add_argument(
        "--base-url", "-b",
        default=BASE_URL,
        help="Base URL of the server (e.g. http://localhost:8080)"
    )
    args = parser.parse_args()

    BASE_URL = args.base_url.rstrip("/")

    try:
        check_health()
        run_tests()
    except Exception as e:
        print("❌ Checker failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
