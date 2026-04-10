import matplotlib.pyplot as plt
import numpy as np

def plot_event(evt_idx, events):
    if evt_idx not in events:
        print(f"Event {evt_idx} not found.")
        return

    data = events[evt_idx]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- bad ---
    if data["bad"]:
        bad_list = [arr for arr in data["bad"] if len(arr) > 0]
        if bad_list:
            bad = np.concatenate(bad_list)
            ax.scatter(
                bad["x"], bad["y"], bad["z"],
                s=10, c="green", label="bad"
            )

    # --- repeat ---
    if data["repeat"]:
        rep_list = [arr for arr in data["repeat"] if len(arr) > 0]
        if rep_list:
            rep = np.concatenate(rep_list)
            ax.scatter(
                rep["x"], rep["y"], rep["z"],
                s=10, c="yellow", label="repeat"
            )

    used_labels = set()

    # --- clusters ---
    for c in data["clusters"]:
        pts = c["points"]
        if len(pts) == 0:
            continue

        ctype = c.get("type", "")
        clabel = c.get("label", "")

        if ctype == "removed":
            color = "black"
            label = "removed"
        elif clabel == "track":
            color = "red"
            label = "track"
        elif clabel == "shower":
            color = "orange"
            label = "shower"
        else:
            color = "blue"
            label = "normal"

        if label not in used_labels:
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=5, c=color, label=label
            )
            used_labels.add(label)
        else:
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=5, c=color
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-65, 65)
    ax.set_ylim(-65, 65)
    ax.set_zlim(-65, 65)
    ax.set_title(f"Event {evt_idx}")

    ax.legend()
    plt.tight_layout()
    plt.show()

def assign_cluster_type(clusters):
    track_idx = 0
    shower_idx = 0
    normal_idx = 0

    for c in clusters:
        if c["label"] == "track":
            c["type"] = f"track_{track_idx}"
            track_idx += 1
        elif c["label"] == "shower":
            c["type"] = f"shower_{shower_idx}"
            shower_idx += 1
        else:
            c["type"] = f"normal_{normal_idx}"
            normal_idx += 1

    return clusters    