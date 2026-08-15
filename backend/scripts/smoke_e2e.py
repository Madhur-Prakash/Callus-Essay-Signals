"""End-to-end smoke test against a running API.

Exercises the real HTTP surface the frontend uses - health, model info, privacy,
analysis of three contrasting essays, persistence round-trip, cache behaviour and
the error paths - and prints a readable report. Intended for local verification
after `uvicorn app.main:app` is up; the pytest suite covers the same ground with
assertions.

Usage
-----
    uv run python -m scripts.smoke_e2e
    uv run python -m scripts.smoke_e2e --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import httpx

# Windows defaults stdout to cp1252 when it is redirected to a file or a pipe, and
# this script prints em-dashes. Without this, `... > out.txt` dies with a
# UnicodeEncodeError that looks exactly like a failing check.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HUMAN = """The robot never worked. That is the honest summary of my sophomore year. I spent seven months on a line-following car that could not follow a line, and I want to explain why that matters to me.

My design was bad from the start. I used two cheap IR sensors mounted too close together, maybe four centimetres apart, because that was what fit on the breadboard I already owned. When the car hit a curve the sensors both read the same value and the controller just guessed. I didn't know the phrase "insufficient sensor separation" then. I only knew that my car drove into a wall in front of forty people at the regional meet.

I rebuilt it. Not immediately, though. I put the whole thing in a shoebox under my bed for about six weeks and told my mom I was done with robotics. Then in January I got bored and pulled it out again, and this time I actually read the sensor datasheet instead of guessing. Four centimetres was wrong. Eleven was better.

The car placed fourth in April. Not first. Fourth. But it finished the course three times out of three, and I still chase that feeling."""

MACHINE = """From an early age, I have been drawn to robotics. What began as a modest interest gradually developed into a genuine commitment. Moreover, the environment in which I worked was demanding, and it required consistent effort. Furthermore, I approached the work methodically, building my understanding one step at a time.

The most significant challenge arose when my initial approach proved inadequate. Additionally, progress was neither linear nor guaranteed, and there were periods of real difficulty. Consequently, I encountered a setback that forced me to reconsider my fundamental assumptions. The obstacle was not merely technical but also personal, testing my resolve.

The turning point came when I decided to rebuild my approach from first principles. Moreover, recognising the limits of my method, I sought guidance and revised my strategy. It required patience, precision, and a willingness to fail. Ultimately, I began to document my process carefully.

The experience instilled in me a deeper appreciation for patience and iteration. Furthermore, I came to understand that meaningful progress depends on perseverance rather than talent. Ultimately, this process cultivated in me a durable capacity for intellectual humility."""

MIXED = """The robot never worked. That is the honest summary of my sophomore year. I spent seven months on a line-following car that could not follow a line, and I want to explain why that matters to me.

My initial design was fundamentally flawed. I employed two inexpensive infrared sensors mounted in close proximity, approximately four centimetres apart, a configuration dictated by the constraints of the breadboard I already possessed. Consequently, when the vehicle encountered a curve, both sensors registered identical values and the controller was left to estimate. Moreover, I was at that time unfamiliar with the concept of insufficient sensor separation.

I rebuilt it. Not immediately, though. I put the whole thing in a shoebox under my bed for about six weeks and told my mom I was done with robotics. Then in January I got bored and pulled it out again, and this time I actually read the sensor datasheet instead of guessing.

The car placed fourth in April. Not first. Fourth. But it finished the course three times out of three, and I still chase that feeling."""

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/") + "/api/v1"

    client = httpx.Client(timeout=180.0)

    # ------------------------------------------------------------- health
    print("\n=== health ===")
    try:
        health = client.get(f"{base}/health").json()
    except Exception as exc:
        print(f"  [{FAIL}] cannot reach the API at {base}: {exc}")
        return 1

    check("health responds", "status" in health, f"status={health['status']}")
    for component in health["components"]:
        state = (
            "available"
            if component["available"]
            else ("disabled" if not component["enabled"] else "UNAVAILABLE")
        )
        print(f"        {component['name']:<18} {state:<12} {component['detail'] or ''}")
    check(
        "detector model is loaded",
        any(c["name"] == "detector_model" and c["available"] for c in health["components"]),
    )
    check("liveness", client.get(f"{base}/health/live").status_code == 200)
    check("readiness", client.get(f"{base}/health/ready").status_code == 200)

    # --------------------------------------------------------- model info
    print("\n=== model info ===")
    info = client.get(f"{base}/model/info").json()
    check("model is ready", bool(info["ready"]))
    print(f"        classifier      {info['document_model']['name']}")
    print(f"        features        {info['document_model']['n_features']}")
    print(f"        calibration     {info['document_model']['calibration']}")
    print(f"        model version   {info['model_version']}")
    print(f"        data regime     {info['data_regime']}")
    print(f"        instrument      {info['language_model'].get('name')}")
    check(
        "methodology states the LM does not classify",
        "never asked to judge authorship"
        in info["methodology"]["what_the_language_model_does"],
    )
    check("feature importance is populated", len(info["feature_importance"]) > 0)

    # ------------------------------------------------------------ privacy
    print("\n=== privacy ===")
    privacy = client.get(f"{base}/essays/privacy").json()
    print(f"        save_essays     {privacy['save_essays_default']}")
    print(f"        retention days  {privacy['retention_days']}")
    check(
        "essay text is never logged",
        any("essay text" in item for item in privacy["what_is_never_logged"]),
    )

    # ----------------------------------------------------------- analyses
    analyses: dict[str, dict[str, Any]] = {}
    for label, text in (("human-style", HUMAN), ("machine-style", MACHINE), ("mixed", MIXED)):
        print(f"\n=== analyse: {label} ===")
        started = time.perf_counter()
        response = client.post(f"{base}/analysis", json={"text": text})
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            check(f"{label} analysed", False, f"HTTP {response.status_code} {response.text[:160]}")
            continue
        body = response.json()
        analyses[label] = body

        print(f"        verdict         {body['label']} ({body['classification']})")
        print(f"        confidence      {body['confidence']} ({body['confidence_score']:.3f})")
        probabilities = ", ".join(f"{k}={v:.3f}" for k, v in body["probabilities"].items())
        print(f"        probabilities   {probabilities}")
        summary = body["summary"]
        print(
            f"        structure       {summary['n_words']} words, "
            f"{summary['n_sentences']} sentences, {summary['n_paragraphs']} paragraphs"
        )
        print(
            f"        sentences       {summary['flagged_sentences']} flagged, "
            f"{summary['uncertain_sentences']} uncertain, "
            f"{summary['human_like_sentences']} human-like"
        )
        statistics = summary["statistics"]
        print(
            f"        key stats       perplexity={statistics['perplexity']:.1f} "
            f"top1={statistics['fraction_top1_tokens']:.3f} "
            f"cv_len={statistics['sentence_length_cv']:.3f} "
            f"burstiness={statistics['burstiness_index']:.3f}"
        )
        print(f"        latency         {elapsed:.0f} ms (server {body['timings'].get('total_ms')} ms)")
        print(f"        persisted       {body['persisted']}   cached: {body['cached']}")

        check(f"{label}: probabilities sum to 1", abs(sum(body["probabilities"].values()) - 1) < 1e-4)
        check(f"{label}: sentence count matches", summary["n_sentences"] == len(body["sentences"]))
        check(f"{label}: rhythm aligns", len(body["rhythm"]) == len(body["sentences"]))
        check(f"{label}: document evidence present", len(body["evidence"]["meters"]) > 0)
        check(f"{label}: no essay text echoed", "text" not in body)

        offsets_ok = all(s["end"] > s["start"] for s in body["sentences"])
        check(f"{label}: sentence offsets valid", offsets_ok)

        flagged = [s for s in body["sentences"] if (s["score"] or 0) >= 0.6]
        with_evidence = [s for s in flagged if s.get("evidence")]
        check(
            f"{label}: flagged sentences carry evidence",
            len(flagged) == len(with_evidence),
            f"{len(with_evidence)}/{len(flagged)}",
        )
        if flagged:
            worst = max(flagged, key=lambda s: s["score"])
            print(f"        most flagged    \"{worst['text'][:70]}...\"  score={worst['score']:.2f}")
            for statement in worst["evidence"]["statements"][:3]:
                print(f"          · {statement}")

        for statement in body["evidence"]["statements"][:3]:
            print(f"        evidence        · {statement}")
        if body["warnings"]:
            print(f"        warnings        {body['warnings'][0][:100]}...")

    # ------------------------------------------------------- directional
    print("\n=== directional check ===")
    if "human-style" in analyses and "machine-style" in analyses:
        human_p = analyses["human-style"]["probabilities"]["human"]
        machine_p = analyses["machine-style"]["probabilities"]["human"]
        check(
            "machine-register scores no more human than the hand-written draft",
            machine_p <= human_p + 0.05,
            f"human={human_p:.3f} vs machine={machine_p:.3f}",
        )

    # ------------------------------------------------------- persistence
    print("\n=== persistence ===")
    if "human-style" in analyses and analyses["human-style"]["persisted"]:
        analysis_id = analyses["human-style"]["analysis_id"]
        fetched = client.get(f"{base}/analysis/{analysis_id}")
        check("stored analysis can be re-fetched", fetched.status_code == 200)
        if fetched.status_code == 200:
            stored = fetched.json()
            check(
                "stored verdict matches",
                stored["classification"] == analyses["human-style"]["classification"],
            )
            sentences = client.get(f"{base}/analysis/{analysis_id}/sentences").json()
            has_offsets = all(s["end"] > s["start"] for s in sentences["sentences"])
            no_text = all(not s.get("text") for s in sentences["sentences"])
            check("stored sentences keep offsets", has_offsets)
            check(
                "stored sentences omit text (SAVE_ESSAYS=false)",
                no_text,
                "offsets only - the client re-slices its own copy",
            )
        listing = client.get(f"{base}/analysis?limit=5")
        check("analysis listing works", listing.status_code == 200,
              f"{listing.json().get('total', 0)} stored")
        deleted = client.delete(f"{base}/analysis/{analysis_id}")
        check("analysis can be deleted", deleted.status_code == 200)
    else:
        print("  [SKIP] MongoDB unavailable or persistence disabled")

    # ------------------------------------------------------------- errors
    print("\n=== error handling ===")
    cases = [
        ("empty essay", {"text": "   "}, 422, "essay_empty"),
        ("too short", {"text": "Way too short to analyse."}, 422, "essay_too_short"),
        ("missing field", {}, 422, "validation_error"),
        ("too long", {"text": "word " * 40000}, 413, None),
    ]
    for name, payload, expected_status, expected_code in cases:
        response = client.post(f"{base}/analysis", json=payload)
        body = response.json()
        code = body.get("error", {}).get("code")
        ok = response.status_code == expected_status and (
            expected_code is None or code == expected_code
        )
        check(f"{name} → {expected_status}", ok, f"got {response.status_code} {code}")
        leaked = any(
            token in response.text.lower() for token in ("traceback", 'file "', "app/services")
        )
        check(f"{name}: no traceback leaked", not leaked)

    check(
        "unknown analysis id → 404/503",
        client.get(f"{base}/analysis/nonexistent-id-xyz").status_code in (404, 503),
    )

    # -------------------------------------------------------- evaluation
    print("\n=== evaluation endpoint ===")
    evaluation = client.get(f"{base}/evaluation").json()
    if evaluation["available"]:
        overall = evaluation["report"]["overall"]
        print(f"        split           {evaluation['report']['split']} (n={overall['n_samples']})")
        print(f"        accuracy        {overall['accuracy']:.4f}")
        print(f"        macro F1        {overall['macro_f1']:.4f}")
        print(f"        human recall    {overall['per_class']['human']['recall']:.4f}")
        print(f"        ECE             {overall['expected_calibration_error']:.4f}")
        check("evaluation report is served", True)
        check("interpretation lines present", len(evaluation["report"]["interpretation"]) > 0)
        check("bias analysis present", "disparity" in evaluation["report"]["bias"])
        if evaluation["failures"]:
            cases_found = len(evaluation["failures"]["confidently_wrong"])
            check("confidently wrong cases present", cases_found >= 3, f"{cases_found} cases")
        check("dataset card present", evaluation["dataset"] is not None)
    else:
        check("evaluation report is served", False, evaluation.get("message", ""))

    # ------------------------------------------------------------- openapi
    print("\n=== openapi ===")
    schema = client.get(args.base_url.rstrip("/") + "/openapi.json").json()
    check("schema documents POST /analysis", "/api/v1/analysis" in schema["paths"])
    check("AnalysisResponse defined", "AnalysisResponse" in schema["components"]["schemas"])

    # -------------------------------------------------------------- summary
    failures = [r for r in results if r[0] == FAIL]
    print("\n" + "=" * 70)
    print(f"  {len(results) - len(failures)}/{len(results)} checks passed")
    if failures:
        print("\n  Failures:")
        for _, name, detail in failures:
            print(f"    - {name} {detail}")
    print("=" * 70)

    client.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
