"""Unit tests for optical health monitoring and vision inference service."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from src.config import InferenceConfig, OpticalHealthConfig, SystemConfig
from src.inference_service import InferenceEngine, InferenceEngineError, InvalidFrameError, OpticalHealthStatus


@pytest.fixture
def system_config() -> SystemConfig:
    """Fixture providing a known test system configuration."""
    return SystemConfig(
        camera_id="line1_overhead_cam01",
        machine_id="press_unit_04",
        optical_health=OpticalHealthConfig(
            blur_laplacian_threshold=100.0,
            dark_frame_mean_threshold=15.0,
            bright_frame_mean_threshold=245.0,
        ),
        inference=InferenceConfig(
            mock_mode=True,
            mock_latency_ms=5.0,
            input_resolution=[224, 224],
        ),
    )


@pytest.fixture
def engine(system_config: SystemConfig) -> InferenceEngine:
    """Fixture providing an initialized InferenceEngine with fixed seed."""
    return InferenceEngine(config=system_config, seed=42)


def create_synthetic_sharp_image(size: int = 224) -> np.ndarray:
    """Generate a high-frequency high-contrast checkerboard image with sharp edges."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    tile_size = 16
    for y in range(0, size, tile_size):
        for x in range(0, size, tile_size):
            if (x // tile_size + y // tile_size) % 2 == 0:
                img[y : y + tile_size, x : x + tile_size] = 255
            else:
                img[y : y + tile_size, x : x + tile_size] = 40
    cv2.line(img, (0, 0), (size, size), (255, 0, 0), 2)
    cv2.line(img, (0, size), (size, 0), (0, 255, 0), 2)
    return img


def test_optical_health_sharp_frame(engine: InferenceEngine) -> None:
    """Test that a high-contrast sharp uint8 frame passes optical health check."""
    sharp_img = create_synthetic_sharp_image()
    status = engine.check_optical_health(sharp_img)

    assert isinstance(status, OpticalHealthStatus)
    assert status.is_valid is True
    assert status.degradation_reason is None
    assert status.laplacian_var > 100.0
    assert 15.0 <= status.mean_brightness <= 245.0


def test_optical_health_float32_normalized_input(engine: InferenceEngine) -> None:
    """Test that a sharp float32 image in range [0.0, 1.0] is properly scaled and passes."""
    sharp_u8 = create_synthetic_sharp_image()
    sharp_f32 = sharp_u8.astype(np.float32) / 255.0
    status = engine.check_optical_health(sharp_f32)

    assert status.is_valid is True
    assert status.degradation_reason is None
    assert status.laplacian_var > 100.0
    assert 15.0 <= status.mean_brightness <= 245.0


def test_optical_health_float32_255_input(engine: InferenceEngine) -> None:
    """Test that a float32 image in range [0.0, 255.0] is handled properly."""
    sharp_u8 = create_synthetic_sharp_image()
    sharp_f32 = sharp_u8.astype(np.float32)
    status = engine.check_optical_health(sharp_f32)

    assert status.is_valid is True
    assert status.laplacian_var > 100.0


def test_optical_health_solid_black_frame(engine: InferenceEngine) -> None:
    """Test that an occluded solid black frame is detected and flagged."""
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    status = engine.check_optical_health(black_img)

    assert status.is_valid is False
    assert status.degradation_reason == "OPTICAL_OCCLUDED_DARK"
    assert status.mean_brightness < 15.0


def test_optical_health_solid_white_frame(engine: InferenceEngine) -> None:
    """Test that an overexposed solid white frame is detected and flagged."""
    white_img = np.full((224, 224, 3), 255, dtype=np.uint8)
    status = engine.check_optical_health(white_img)

    assert status.is_valid is False
    assert status.degradation_reason == "OPTICAL_OCCLUDED_BRIGHT"
    assert status.mean_brightness > 245.0


def test_optical_health_blurred_frame(engine: InferenceEngine) -> None:
    """Test that an intentionally blurred frame is flagged as OPTICAL_BLURRED."""
    sharp_img = create_synthetic_sharp_image()
    blurred_img = cv2.GaussianBlur(sharp_img, (31, 31), sigmaX=15.0)
    status = engine.check_optical_health(blurred_img)

    assert status.is_valid is False
    assert status.degradation_reason == "OPTICAL_BLURRED"
    assert status.laplacian_var < 100.0


def test_optical_health_grayscale_input(engine: InferenceEngine) -> None:
    """Test optical health check accepts single-channel 2D grayscale frames."""
    sharp_color = create_synthetic_sharp_image()
    gray = cv2.cvtColor(sharp_color, cv2.COLOR_BGR2GRAY)
    status = engine.check_optical_health(gray)

    assert status.is_valid is True
    assert status.laplacian_var > 100.0


def test_optical_health_3d_single_channel_input(engine: InferenceEngine) -> None:
    """Test optical health check accepts (H, W, 1) single-channel frames."""
    sharp_color = create_synthetic_sharp_image()
    gray = cv2.cvtColor(sharp_color, cv2.COLOR_BGR2GRAY)
    gray_3d = np.expand_dims(gray, axis=2)
    status = engine.check_optical_health(gray_3d)

    assert status.is_valid is True
    assert status.laplacian_var > 100.0


def test_optical_health_unsupported_channels(engine: InferenceEngine) -> None:
    """Test that frames with unsupported channel count raise InvalidFrameError."""
    frame_5ch = np.zeros((224, 224, 5), dtype=np.uint8)
    with pytest.raises(InvalidFrameError, match="Unsupported number of image channels"):
        engine.check_optical_health(frame_5ch)


def test_invalid_frame_exceptions(engine: InferenceEngine) -> None:
    """Test that malformed inputs, empty arrays, and None raise InvalidFrameError."""
    with pytest.raises(InvalidFrameError, match="non-null"):
        engine.check_optical_health(None)  # type: ignore

    with pytest.raises(InvalidFrameError, match="empty array"):
        engine.check_optical_health(np.array([]))

    with pytest.raises(InvalidFrameError, match="dimensions"):
        engine.check_optical_health(np.zeros((10, 10, 10, 10), dtype=np.uint8))


def test_deterministic_inference_output(system_config: SystemConfig) -> None:
    """Test that two engines initialized with the same seed yield identical outputs."""
    eng1 = InferenceEngine(config=system_config, seed=123)
    eng2 = InferenceEngine(config=system_config, seed=123)
    sharp_img = create_synthetic_sharp_image()

    res1 = eng1.run_inference(sharp_img, inject_anomaly=False)
    res2 = eng2.run_inference(sharp_img, inject_anomaly=False)

    assert res1.vision_score == res2.vision_score
    assert res1.latency_ms > 0
    assert np.allclose(res1.heatmap, res2.heatmap)


def test_mock_inference_normal_frame(engine: InferenceEngine) -> None:
    """Test mock inference returns scores within [0.0, 1.0] and valid data contract."""
    sharp_img = create_synthetic_sharp_image()
    result = engine.run_inference(sharp_img, inject_anomaly=False)

    assert 0.0 <= result.vision_score <= 1.0
    assert result.is_blurred is False
    assert result.is_occluded is False
    assert result.latency_ms >= 4.0
    assert result.heatmap is not None
    assert result.heatmap.shape == (224, 224)
    assert 0.0 <= float(np.min(result.heatmap)) <= float(np.max(result.heatmap)) <= 1.0
    assert result.camera_id == "line1_overhead_cam01"
    assert result.model_metadata["engine"] == "mock"
    assert result.optical_health.is_valid is True
    assert "T" in result.timestamp_utc and result.timestamp_utc.endswith("Z")


def test_mock_inference_anomalous_frame(engine: InferenceEngine) -> None:
    """Test mock inference with injected defect yields elevated anomaly scores."""
    sharp_img = create_synthetic_sharp_image()
    result = engine.run_inference(sharp_img, inject_anomaly=True)

    assert 0.70 <= result.vision_score <= 1.0
    assert result.is_blurred is False
    assert result.is_occluded is False
    assert result.heatmap is not None
    assert result.metadata["inject_anomaly"] is True


def test_mock_inference_on_degraded_frame(engine: InferenceEngine) -> None:
    """Test that running inference on an occluded frame triggers degraded mode."""
    dark_img = np.zeros((224, 224, 3), dtype=np.uint8)
    result = engine.run_inference(dark_img)

    assert result.is_occluded is True
    assert result.is_blurred is False
    assert result.vision_score == 0.0
    assert result.heatmap is None
    assert result.optical_health.is_valid is False
    assert result.metadata["inference_mode"] == "degraded_optical_bypass"


def test_onnx_mode_initialization_missing_model() -> None:
    """A configured model that does not exist is an error, never a silent switch to synthetic scores."""
    cfg = SystemConfig(
        inference=InferenceConfig(backend="onnx", score_reference=0.5, model_path="nonexistent_model.onnx")
    )
    with pytest.raises(InferenceEngineError):
        InferenceEngine(config=cfg)


def test_backend_without_reference_is_rejected() -> None:
    with pytest.raises(ValueError):
        InferenceConfig(backend="patchcore", model_path="m.pt")


def test_distance_normalization_maps_reference_to_half() -> None:
    cfg = SystemConfig(inference=InferenceConfig(backend="onnx", score_reference=2.0, model_path="m.onnx"))
    with patch.object(InferenceEngine, "_init_onnx_session", lambda self, p: None):
        eng = InferenceEngine(config=cfg)
    assert eng.normalize_distance(2.0) == pytest.approx(0.5)
    assert eng.normalize_distance(10.0) == 1.0 and eng.normalize_distance(0.0) == 0.0


def test_onnx_mode_successful_session_init(tmp_path: Path) -> None:
    """Test successful ONNX session initialization when file exists and ort creates session."""
    dummy_model_file = tmp_path / "model.onnx"
    dummy_model_file.write_bytes(b"dummy onnx content")

    mock_sess = MagicMock()
    inp = MagicMock()
    inp.name = "input_tensor"
    out = MagicMock()
    out.name = "output_tensor"
    mock_sess.get_inputs.return_value = [inp]
    mock_sess.get_outputs.return_value = [out]

    with patch("onnxruntime.InferenceSession", return_value=mock_sess):
        cfg = SystemConfig(
            inference=InferenceConfig(backend="onnx", score_reference=0.5, model_path=str(dummy_model_file))
        )
        eng = InferenceEngine(config=cfg)
        assert eng._onnx_session is not None
        assert eng._input_name == "input_tensor"
        assert eng._output_name == "output_tensor"


def test_onnx_mode_session_init_failure_raises_error(tmp_path: Path) -> None:
    """Test that failed ONNX session creation raises InferenceEngineError."""
    dummy_model_file = tmp_path / "corrupt_model.onnx"
    dummy_model_file.write_bytes(b"corrupt")

    with patch("onnxruntime.InferenceSession", side_effect=Exception("Invalid model protobuf")):
        cfg = SystemConfig(
            inference=InferenceConfig(backend="onnx", score_reference=0.5, model_path=str(dummy_model_file))
        )
        with pytest.raises(InferenceEngineError, match="ONNX initialization failed"):
            InferenceEngine(config=cfg)


def test_onnx_forward_execution_mocked_session() -> None:
    """Test ONNX forward pass execution with mocked onnxruntime session."""
    cfg = SystemConfig(
        inference=InferenceConfig(backend="onnx", score_reference=0.5, model_path="dummy.onnx")
    )
    with patch.object(InferenceEngine, "_init_onnx_session", lambda self, p: None):
        eng = InferenceEngine(config=cfg)

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input_tensor"
    mock_output = MagicMock()
    mock_output.name = "output_score"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.array([[0.82]], dtype=np.float32)]

    eng._onnx_session = mock_session
    eng._input_name = "input_tensor"
    eng._output_name = "output_score"

    sharp_img = create_synthetic_sharp_image()
    res = eng.run_inference(sharp_img)

    assert res.vision_score == pytest.approx(0.82, abs=1e-3)
    assert res.model_metadata["engine"] == "onnxruntime"
    assert res.metadata["inference_mode"] == "onnx"


def test_onnx_forward_execution_patch_heatmap() -> None:
    """Test ONNX forward pass execution when model returns patch anomaly heatmap."""
    cfg = SystemConfig(
        inference=InferenceConfig(backend="onnx", score_reference=0.5, model_path="dummy.onnx")
    )
    with patch.object(InferenceEngine, "_init_onnx_session", lambda self, p: None):
        eng = InferenceEngine(config=cfg)

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input_tensor"
    mock_output = MagicMock()
    mock_output.name = "heatmap_tensor"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]

    dummy_heatmap = np.zeros((1, 1, 224, 224), dtype=np.float32)
    dummy_heatmap[0, 0, 100:120, 100:120] = 0.88
    mock_session.run.return_value = [dummy_heatmap]

    eng._onnx_session = mock_session
    eng._input_name = "input_tensor"
    eng._output_name = "heatmap_tensor"

    # Pass sharp grayscale float image
    sharp_img = create_synthetic_sharp_image()
    sharp_gray_float = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    res = eng.run_inference(sharp_gray_float)

    assert res.vision_score == pytest.approx(0.88, abs=1e-3)
    assert res.heatmap is not None
    assert res.heatmap.shape == (224, 224)


def test_onnx_forward_execution_failure_raises_error() -> None:
    """Test that runtime exceptions in ONNX session.run raise InferenceEngineError."""
    cfg = SystemConfig(
        inference=InferenceConfig(backend="onnx", score_reference=0.5, model_path="dummy.onnx")
    )
    with patch.object(InferenceEngine, "_init_onnx_session", lambda self, p: None):
        eng = InferenceEngine(config=cfg)

    mock_session = MagicMock()
    mock_session.run.side_effect = RuntimeError("Hardware execution failed")
    eng._onnx_session = mock_session
    eng._input_name = "in"
    eng._output_name = "out"

    sharp_img = create_synthetic_sharp_image()
    with pytest.raises(InferenceEngineError, match="ONNX inference failure"):
        eng.run_inference(sharp_img)