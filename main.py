"""PixelBridge entry point (Visual Studio startup file).

    python main.py                         open the GUI
    python main.py gui [IMAGE]             open the GUI (optionally with an image)
    python main.py a  PHOTO OUT.png  [--width 64 --colors 16 --dither fs ...]
    python main.py b  SPRITE OUT.png [--scale 4 --method ours --mode palette]
    python main.py roundtrip PHOTO OUTDIR [--width 64 --scale 4 --method ours]
    python main.py chara PHOTO OUT.png [--bg LANDMARK.jpg --bangs full --hair short --shared-palette]

Run `python main.py a -h` (or b, roundtrip) for all options.
"""
from __future__ import annotations

import argparse
import os
import sys

from pixelbridge.core.io import read_image, write_image
from pixelbridge.module_a.pipeline import PixelArtParams, photo_to_pixel_art
from pixelbridge.module_b.pipeline import METHODS, upscale


def add_a_options(p: argparse.ArgumentParser) -> None:
    d = PixelArtParams()
    p.add_argument("--width", type=int, default=d.out_width, help="pixel art width (32-128)")
    p.add_argument("--colors", type=int, default=d.n_colors, help="number of colors (4-32)")
    p.add_argument("--palette", default=d.palette, help="auto or a fixed palette name")
    p.add_argument("--palette-method", default=d.palette_method, choices=["median_cut", "kmeans"])
    p.add_argument("--downsample", default=d.downsample, choices=["block", "edge"])
    p.add_argument("--dither", default=d.dither, choices=["none", "fs", "bayer"])
    p.add_argument("--strength", type=float, default=d.dither_strength, help="dither strength 0-1")
    p.add_argument("--enhance", type=float, default=d.enhance, help="contrast/saturation boost 0-1")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--no-outline", action="store_true")
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument("--save-steps", metavar="DIR", help="also save intermediate images")


def params_from_args(args) -> PixelArtParams:
    return PixelArtParams(
        out_width=args.width, n_colors=args.colors, palette=args.palette,
        palette_method=args.palette_method, downsample=args.downsample,
        dither=args.dither, dither_strength=args.strength, enhance=args.enhance,
        smooth=not args.no_smooth, outline=not args.no_outline, cleanup=not args.no_cleanup)


def run_a(args) -> None:
    result = photo_to_pixel_art(read_image(args.input), params_from_args(args))
    write_image(args.output, result.image)
    if args.save_steps:
        for name, img in result.steps.items():
            write_image(os.path.join(args.save_steps, f"{name}.png"), img)
    times = ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items())
    print(f"saved {args.output}  {result.image.shape[1]}x{result.image.shape[0]}, "
          f"{len(result.palette)} colors  [{times}]")


def run_b(args) -> None:
    out = upscale(read_image(args.input), args.scale, args.method, args.mode)
    write_image(args.output, out)
    print(f"saved {args.output}  {out.shape[1]}x{out.shape[0]}")


def run_roundtrip(args) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    photo = read_image(args.input)
    result = photo_to_pixel_art(photo, params_from_args(args))
    big = upscale(result.image, args.scale, args.method, args.mode)
    write_image(os.path.join(args.outdir, "0_photo.png"), photo)
    write_image(os.path.join(args.outdir, "1_pixel_art.png"), result.image)
    write_image(os.path.join(args.outdir, f"2_upscaled_{args.method}_x{args.scale}.png"), big)
    print(f"round trip saved in {args.outdir}")


def run_chara(args) -> None:
    from pixelbridge.character import compose_scene, default_background, make_character, pixel_background
    box = tuple(args.head) if args.head else None
    auto = lambda v: None if v == "auto" else v  # noqa: E731
    flag = {"auto": None, "yes": True, "no": False}
    character = make_character(read_image(args.input), head_box=box,
                               length=auto(args.hair), bangs=auto(args.bangs),
                               smile=flag[args.smile], happy_eyes=flag[args.happy_eyes])
    if args.bg:
        background, palette = pixel_background(read_image(args.bg), width=args.bg_width, n_colors=args.bg_colors)
    else:
        background, palette = default_background(), None
    scene = compose_scene(background, character, x=args.x, ground=args.ground,
                          palette=palette if args.shared_palette else None)
    big = upscale(scene, args.scale, args.method, "palette")
    write_image(args.output, big)
    stem, ext = os.path.splitext(args.output)
    write_image(f"{stem}_sprite{ext or '.png'}", character.image)
    a = character.attrs
    print(f"saved {args.output}  {big.shape[1]}x{big.shape[0]}")
    print(f"  head box {' '.join(map(str, a.box))} | hair {a.length}, bangs {a.bangs}, part {a.part:.2f} | "
          f"smile {'yes' if a.smile else 'no'}, happy eyes {'yes' if a.happy_eyes else 'no'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="PixelBridge", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    g = sub.add_parser("gui", help="open the GUI")
    g.add_argument("input", nargs="?")

    a = sub.add_parser("a", help="Module A: photo -> pixel art")
    a.add_argument("input")
    a.add_argument("output")
    add_a_options(a)

    b = sub.add_parser("b", help="Module B: pixel art -> high resolution")
    b.add_argument("input")
    b.add_argument("output")
    b.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    b.add_argument("--method", default="ours", choices=METHODS)
    b.add_argument("--mode", default="palette", choices=["palette", "smooth"])

    r = sub.add_parser("roundtrip", help="photo -> pixel art -> high resolution")
    r.add_argument("input")
    r.add_argument("outdir")
    add_a_options(r)
    r.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    r.add_argument("--method", default="ours", choices=METHODS)
    r.add_argument("--mode", default="palette", choices=["palette", "smooth"])

    c = sub.add_parser("chara", help="photo -> chibi pixel character on a pixel background")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--head", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                   help="head box in the photo (default: found from skin color)")
    c.add_argument("--hair", default="auto", choices=["auto", "short", "medium", "long"])
    c.add_argument("--bangs", default="auto", choices=["auto", "none", "curtain", "full", "side"])
    c.add_argument("--smile", default="auto", choices=["auto", "yes", "no"])
    c.add_argument("--happy-eyes", default="auto", choices=["auto", "yes", "no"])
    c.add_argument("--bg", help="your own landmark photo for the background (default: built-in Namsan scene)")
    c.add_argument("--bg-width", type=int, default=128)
    c.add_argument("--bg-colors", type=int, default=24)
    c.add_argument("--x", type=float, default=0.25, help="horizontal position of the character, 0-1")
    c.add_argument("--ground", type=float, default=0.92, help="where the feet stand, 0 (top) - 1 (bottom)")
    c.add_argument("--shared-palette", action="store_true", help="re-color the character with the background palette")
    c.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    c.add_argument("--method", default="scale2x", choices=METHODS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in (None, "gui"):
            from pixelbridge.gui.app import main as gui_main
            gui_main(getattr(args, "input", None))
        elif args.command == "a":
            run_a(args)
        elif args.command == "b":
            run_b(args)
        elif args.command == "chara":
            run_chara(args)
        else:
            run_roundtrip(args)
    except NotImplementedError as e:
        print(f"Not implemented yet: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
