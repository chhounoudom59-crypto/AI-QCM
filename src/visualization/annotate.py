from __future__ import annotations

import cv2
import numpy as np


def put_status_banner(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (30, 30, 30), -1)
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_detection_boxes(image: np.ndarray, detections: list, show_labels: bool = True, show_confidence: bool = True) -> np.ndarray:
    """Draw bounding boxes for all detections.
    
    Args:
        image: Input image (BGR)
        detections: List of BlockDetection objects
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
    
    Returns:
        Annotated image
    """
    out = image.copy()
    colors = {
        "mcq_block": (0, 255, 0),
        "roman_block": (255, 0, 0),
        "tfng_block": (0, 255, 255),
        "completion_block": (255, 0, 255),
    }
    
    for det in detections:
        color = colors.get(det.label, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        
        # Draw label
        if show_labels or show_confidence:
            text = det.label if show_labels else ""
            if show_confidence:
                text += f" {det.confidence:.2f}" if text else f"{det.confidence:.2f}"
            
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            bg_x1 = det.x1
            bg_y1 = max(10, det.y1 - text_size[1] - 10)
            bg_x2 = bg_x1 + text_size[0] + 10
            bg_y2 = bg_y1 + text_size[1] + 10
            
            cv2.rectangle(out, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.putText(out, text, (bg_x1 + 5, bg_y1 + text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    return out


def draw_answer_overlay(image: np.ndarray, answers: list[tuple[str, str, float]], font_scale: float = 0.7) -> np.ndarray:
    """Draw extracted answers on image.
    
    Args:
        image: Input image
        answers: List of (question_id, answer, confidence) tuples
        font_scale: Font size
    
    Returns:
        Annotated image
    """
    out = image.copy()
    h, w = out.shape[:2]
    
    # Draw semi-transparent background for text
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 150), (300, h), (0, 0, 0), -1)
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    
    # Draw answers
    y_offset = h - 120
    for qid, answer, conf in answers[:10]:  # Show first 10
        text = f"{qid}: {answer} ({conf:.2f})"
        cv2.putText(out, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
        y_offset -= 15
    
    return out


def draw_score_report(image: np.ndarray, metrics: dict, position: tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw score metrics on image.
    
    Args:
        image: Input image
        metrics: Scoring metrics dict
        position: (x, y) top-left position for text
    
    Returns:
        Annotated image
    """
    out = image.copy()
    
    score_text = [
        f"Total: {metrics.get('total_questions', 0)}",
        f"Correct: {metrics.get('correct', 0)}",
        f"Wrong: {metrics.get('wrong', 0)}",
        f"Score: {metrics.get('percentage', 0):.1f}%",
    ]
    
    x, y = position
    for i, text in enumerate(score_text):
        cv2.putText(out, text, (x, y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    return out
