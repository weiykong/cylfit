import json
from pathlib import Path

from cylinderfit2026.benchmarks import available_methods, benchmark_markdown_report, run_benchmark, summarize_benchmark


def main() -> None:
    print("methods:", ", ".join(available_methods()))
    raw_results = run_benchmark()
    results = [result.to_dict() for result in raw_results]
    print(json.dumps({"summary": summarize_benchmark(raw_results), "results": results}, indent=2))
    out = Path(__file__).with_name("benchmark_report.md")
    out.write_text(benchmark_markdown_report(raw_results), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
