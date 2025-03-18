from copy import deepcopy


def convert_real_time_anns_to_datapoint(
    anns: dict[str, dict],
) -> list[tuple[str, list[dict]]]:
    data: list[tuple[str, list[dict]]] = []
    for video, dialogue in anns.items():
        i = 0
        is_eval = False
        while i < len(dialogue):
            if not is_eval and dialogue[i]["eval"]:
                is_eval = True
            if is_eval and not dialogue[i]["eval"]:
                is_eval = False
                data.append((video, deepcopy(dialogue[:i])))
                # set eval to False for the added utterances
                # as they will be used as part of the context for the next data point.
                for utter in dialogue[:i]:
                    utter["eval"] = False
            i += 1
        # take care of the stragglers
        if is_eval:
            data.append((video, deepcopy(dialogue[:i])))
    return data
