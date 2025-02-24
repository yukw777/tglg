import json


def main(holo_assist_anns_file: str, output_file: str) -> None:
    with open(holo_assist_anns_file) as f:
        holo_assist_anns = json.load(f)

    anns = {}
    for holo_assist_ann in holo_assist_anns:
        ann: list[list[dict]] = []
        i, j = 0, 0
        while i < len(holo_assist_ann["events"]):
            correction_event = holo_assist_ann["events"][i]
            # look for a correction
            if correction_event["label"] == "Conversation" and correction_event[
                "attributes"
            ]["Conversation Purpose"].endswith("correct the wrong action"):
                # look back and find the first `instruction`
                j = i - 1
                while j >= 0:
                    instruction_event = holo_assist_ann["events"][j]
                    if instruction_event[
                        "label"
                    ] == "Conversation" and instruction_event["attributes"][
                        "Conversation Purpose"
                    ].endswith("instruction"):
                        events = [
                            event
                            for event in holo_assist_ann["events"][j : i + 1]
                            if event["label"] == "Conversation"
                            # filter out student conversations
                            and event["attributes"]["Conversation Purpose"].startswith(
                                "instructor"
                            )
                        ]
                        # include any conversations within 10 seconds of the last event, i.e., correct the wrong action.
                        # since interesting dialogues happen after a wrong action is corrected.
                        k = i + 1
                        while (
                            k < len(holo_assist_ann["events"])
                            and holo_assist_ann["events"][k]["start"]
                            < events[0]["start"] + 10
                        ):
                            event = holo_assist_ann["events"][k]
                            if event["label"] == "Conversation" and event["attributes"][
                                "Conversation Purpose"
                            ].startswith("instructor"):
                                events.append(event)
                            k += 1

                        # check if these events overlap with previous.
                        # If they overlap, it just means the correction does not have its own
                        # instruction
                        if len(ann) > 0 and events[0]["start"] < ann[-1][-1]["end"]:
                            # overlap, merge with previous
                            # first find where ann[-1][-1]['end'] is in events
                            for k, event in enumerate(events):
                                if event["end"] == ann[-1][-1]["end"]:
                                    ann[-1].extend(events[k + 1 :])
                                    break
                        else:
                            ann.append(events)
                        break
                    j -= 1
            i += 1

        if len(ann) > 0:
            anns[holo_assist_ann["video_name"]] = ann

    with open(output_file, "w") as f:
        json.dump(anns, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--holo_assist_anns_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    main(args.holo_assist_anns_file, args.output_file)
