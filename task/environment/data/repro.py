"""Smoke check over the shipped fixtures."""

import json
import pathlib

from sfttrainer.pipeline import build_run

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    conversations = json.loads((ROOT / "fixtures" / "conversations.json").read_text())
    config = json.loads((ROOT / "fixtures" / "config.json").read_text())
    run = build_run(conversations, config)

    cap = config["max_tokens_per_microbatch"]
    for index, batch in enumerate(run["microbatches"]):
        rows, width = batch["tokens"].shape
        padded = rows * width
        if padded > cap:
            raise AssertionError(
                f"micro-batch {index} holds {rows} rows of width {width} "
                f"= {padded} padded tokens, over the budget of {cap}"
            )

    print(f"ok: {len(run['examples'])} examples, "
          f"{len(run['microbatches'])} micro-batches, {len(run['steps'])} steps")


if __name__ == "__main__":
    main()
