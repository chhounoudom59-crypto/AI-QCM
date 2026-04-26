from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import OMRPipeline
from src.visualization.annotate import draw_detection_boxes, put_status_banner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OMR pipeline inference on a single sheet image")
    parser.add_argument("--image", required=True, help="Path to input sheet image")
    parser.add_argument("--yolo", default="artifacts/yolo/best.pt", help="Path to YOLO model")
    parser.add_argument("--out", default="outputs/infer_result.jpg", help="Output path for annotated image")
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = OMRPipeline(args.yolo)
    
    # Process image
    print(f"Processing: {args.image}")
    result = pipeline.process_image(args.image)
    
    # Print results
    print(f"\nDetections: {len(result.detections_raw)}")
    for i, det in enumerate(result.detections_raw):
        print(f"  Block {i}: {det.label} (confidence: {det.confidence:.2f})")
    
    print(f"\nExtracted answers:")
    for qid, answer, conf in result.extracted_answers:
        print(f"  {qid}: {answer} ({conf:.2f})")
    
    # Draw annotations
    vis = draw_detection_boxes(result.image_detections, result.detections_raw, show_labels=True, show_confidence=True)
    banner = f"Detections: {len(result.detections_raw)}"
    vis = put_status_banner(vis, banner)

    # Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
