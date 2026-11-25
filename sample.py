import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a contact sheet of a PointMaze trajectory."
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode number to inspect (default: 0)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/point_maze"),
        help="PointMaze data directory (default: data/point_maze)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Output image path (default: pointmaze_episode_<episode>_contact_sheet.png)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=10,
        help="Number of images per row (default: 10)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an image window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    obs_dir = args.data_dir / "obses"
    episode_path = obs_dir / f"episode_{args.episode:03d}.pth"

    if not episode_path.is_file():
        raise FileNotFoundError(
            f"Episode file not found: {episode_path}\n"
            f"Available episode count: {len(list(obs_dir.glob('episode_*.pth')))}"
        )

    trajectory = torch.load(episode_path, map_location="cpu")
    print(f"Episode file: {episode_path}")
    print(f"Trajectory shape: {tuple(trajectory.shape)}")
    print(f"Data type: {trajectory.dtype}")

    if trajectory.ndim != 4 or trajectory.shape[-1] != 3:
        raise ValueError(
            f"Expected THWC RGB observations, but got shape {tuple(trajectory.shape)}"
        )

    if args.columns < 1:
        raise ValueError("--columns must be at least 1")

    frame_count = trajectory.shape[0]
    columns = min(args.columns, frame_count)
    rows = (frame_count + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2 * columns, 2 * rows),
        squeeze=False,
    )
    for frame_index, ax in enumerate(axes.flat):
        if frame_index < frame_count:
            ax.imshow(trajectory[frame_index].numpy())
            ax.set_title(str(frame_index), fontsize=8)
        ax.axis("off")
    fig.suptitle(f"PointMaze - Episode {args.episode} ({frame_count} frames)")
    fig.tight_layout()

    output_path = args.save or Path(
        f"pointmaze_episode_{args.episode:03d}_contact_sheet.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Saved {frame_count}-frame contact sheet to: {output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
