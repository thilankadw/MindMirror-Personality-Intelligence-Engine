"""Inference pipeline for agreeableness."""
from pipelines.personality.run_trait_inference import (
    build_inference_parser,
    run_trait_inference,
)

TRAIT = "agreeableness"


def run_inference_pipeline(
    posts,
    author=None,
    model_path=None,
    output_path=None,
    include_features=False,
    gap_weeks=0.0,
):
    """Run inference pipeline."""
    return run_trait_inference(
        trait=TRAIT,
        posts=posts,
        author=author,
        model_path=model_path,
        output_path=output_path,
        include_features=include_features,
        gap_weeks=gap_weeks,
    )


def main():
    """Run the module entry point."""
    parser = build_inference_parser()
    args = parser.parse_args()
    run_inference_pipeline(
        posts=args.posts,
        author=args.author,
        model_path=args.model_path,
        output_path=args.output,
        include_features=args.show_features,
        gap_weeks=args.gap_weeks,
    )


if __name__ == "__main__":
    main()
