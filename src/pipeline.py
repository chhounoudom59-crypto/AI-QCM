from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np
from pathlib import Path

from src.preprocessing.image_ops import load_image
from src.preprocessing.contrast import preprocess_for_inference
from src.alignment.perspective import perspective_correct
from src.detection.yolo_layout import YoloLayoutDetector, crop_detection
from src.detection.extractors import route_and_extract, ExtractionResult
from src.mapping.answer_map import AnswerItem
from src.scoring.compare import compute_metrics


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    image_aligned: np.ndarray
    image_detections: np.ndarray
    detections_raw: list = field(default_factory=list)
    extracted_answers: list[tuple[str, str, float]] = field(default_factory=list)  # (question_id, answer, confidence)
    answer_map: dict[str, str] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    debug_info: dict = field(default_factory=dict)


class OMRPipeline:
    """Main orchestrator for the entire OMR pipeline."""
    
    def __init__(self, model_path: str):
        """Initialize pipeline with YOLO model path."""
        self.model_path = model_path
        self.detector = YoloLayoutDetector(model_path)
        self.debug = False
    
    def process_image(
        self,
        image_input: str | np.ndarray,
        question_mapping: dict[str, dict] | None = None,
    ) -> PipelineResult:
        """Process single OMR sheet end-to-end.
        
        Args:
            image_input: File path or numpy array (BGR)
            question_mapping: Map block_id to question_id and options
                Example: {"block_0": {"question_id": "Q1", "options": ["A", "B", "C", "D"]}}
        
        Returns:
            Complete pipeline result
        """
        # Layer 1: Input Acquisition
        if isinstance(image_input, str):
            image_original = load_image(image_input)
        else:
            image_original = image_input.copy()
        
        result = PipelineResult()
        result.debug_info["step1_input_shape"] = image_original.shape
        
        # Layer 2: Preprocessing
        image_preprocessed, original_size = preprocess_for_inference(image_original)
        result.debug_info["step2_preprocessed_shape"] = image_preprocessed.shape
        
        # Layer 3: Geometric Normalization (Alignment)
        image_aligned = perspective_correct(image_original)
        result.image_aligned = image_aligned
        result.debug_info["step3_aligned_shape"] = image_aligned.shape
        
        # Layer 4: AI Detection Layer
        detections = self.detector.detect(image_aligned, conf=0.25)
        result.detections_raw = detections
        result.debug_info["step4_detections_count"] = len(detections)
        result.debug_info["step4_detection_labels"] = [d.label for d in detections]
        
        # Layer 5: Region Cropping & Routing
        # Layer 6: Region-Level Extraction (Core Logic)
        extracted_answers = []
        for block_idx, det in enumerate(detections):
            roi = crop_detection(image_aligned, det)
            
            # Get configuration for this block if available
            block_key = f"block_{block_idx}"
            block_config = question_mapping.get(block_key, {}) if question_mapping else {}
            
            # Route and extract
            extraction_result = route_and_extract(roi, det.label, block_config)
            
            question_id = block_config.get("question_id", f"Q{block_idx+1}")
            extracted_answers.append((question_id, extraction_result.answer, extraction_result.confidence))
            
            if self.debug:
                print(f"Block {block_idx} ({det.label}): Q{question_id} = {extraction_result.answer} ({extraction_result.confidence:.2f})")
        
        result.extracted_answers = extracted_answers
        
        # Layer 7: Answer Mapping
        result.answer_map = {qid: ans for qid, ans, _ in extracted_answers}
        
        # Visualization (optional)
        image_vis = image_aligned.copy()
        for det in detections:
            cv2.rectangle(image_vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            cv2.putText(
                image_vis,
                f"{det.label}:{det.confidence:.2f}",
                (det.x1, max(20, det.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        result.image_detections = image_vis
        
        result.debug_info["step7_final_answer_map"] = result.answer_map
        
        return result
    
    def process_sheet_comparison(
        self,
        student_image_input: str | np.ndarray,
        teacher_answer_map: dict[str, str],
        question_mapping: dict[str, dict] | None = None,
    ) -> dict:
        """Process student sheet and compare against teacher answers.
        
        Args:
            student_image_input: Student sheet image
            teacher_answer_map: Ground truth answers {question_id: answer}
            question_mapping: Optional block configuration
        
        Returns:
            Dictionary with scores and metrics
        """
        # Process student sheet
        result = self.process_image(student_image_input, question_mapping)
        student_map = result.answer_map
        
        # Compare
        metrics = compute_metrics(student_map, teacher_answer_map)
        result.metrics = metrics
        
        return {
            "student_answers": student_map,
            "teacher_answers": teacher_answer_map,
            "metrics": metrics,
            "debug": result.debug_info,
            "image_aligned": result.image_aligned,
            "image_detections": result.image_detections,
        }
    
    def enable_debug(self):
        """Enable debug output."""
        self.debug = True
