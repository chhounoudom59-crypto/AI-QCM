"""
Example: Basic OMR sheet inference with detection and extraction.
"""

from pathlib import Path
import cv2
from src.pipeline import OMRPipeline


def main():
    # Configuration
    MODEL_PATH = "artifacts/yolo/best.pt"
    SAMPLE_IMAGE = "sample_sheet.jpg"  # Path to your test image
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = OMRPipeline(MODEL_PATH)
    pipeline.enable_debug()
    
    # Optional: Define question mapping
    # This maps YOLO detections to question IDs and options
    question_mapping = {
        "block_0": {
            "question_id": "Q1",
            "options": ["A", "B", "C", "D"],
        },
        "block_1": {
            "question_id": "Q2",
            "options": ["A", "B", "C", "D"],
        },
        # Add more as needed...
    }
    
    # Process image
    print(f"Processing: {SAMPLE_IMAGE}")
    result = pipeline.process_image(SAMPLE_IMAGE, question_mapping=question_mapping)
    
    # Display results
    print(f"\nDetections found: {len(result.detections_raw)}")
    for i, det in enumerate(result.detections_raw):
        print(f"  Block {i}: {det.label} (confidence: {det.confidence:.2f})")
    
    print(f"\nExtracted answers:")
    for qid, answer, conf in result.extracted_answers:
        print(f"  {qid}: {answer} ({conf:.2f})")
    
    print(f"\nFinal answer map:")
    print(result.answer_map)
    
    # Save visualizations
    cv2.imwrite(str(OUTPUT_DIR / "aligned.jpg"), result.image_aligned)
    cv2.imwrite(str(OUTPUT_DIR / "detections.jpg"), result.image_detections)
    print(f"\nOutputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
