import os
import pytest
from tests.mock_mvtec import generate_mock_category
from src.mvtec import MVTecTrainNormal, MVTecTest


def test_mock_dataset_generation(tmp_path):
    data_root = str(tmp_path / "data" / "mvtec_ad")
    generate_mock_category(data_root, "bottle", num_train=5, num_test=4)

    train_dir = os.path.join(data_root, "bottle", "train", "good")
    test_good_dir = os.path.join(data_root, "bottle", "test", "good")
    test_defect_dir = os.path.join(data_root, "bottle", "test", "defect")
    gt_defect_dir = os.path.join(data_root, "bottle", "ground_truth", "defect")

    assert os.path.exists(train_dir)
    assert len(os.listdir(train_dir)) == 5

    assert os.path.exists(test_defect_dir)
    assert os.path.exists(gt_defect_dir)
    assert len(os.listdir(test_defect_dir)) == 2
    assert len(os.listdir(gt_defect_dir)) == 2

    # Instantiate datasets and check loading
    train_ds = MVTecTrainNormal(data_root, "bottle")
    assert len(train_ds) == 5

    test_ds = MVTecTest(data_root, "bottle")
    assert len(test_ds) == 4

    x, y, mask, meta = test_ds[0]
    assert x.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)


def test_missing_directory_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        MVTecTrainNormal(str(tmp_path), "non_existent_category")
    assert "Training directory not found" in str(exc_info.value)
    assert "python scripts/download_dataset.py" in str(exc_info.value)

    with pytest.raises(FileNotFoundError) as exc_info2:
        MVTecTest(str(tmp_path), "non_existent_category")
    assert "Test directory not found" in str(exc_info2.value)


def test_official_counts_detect_non_official_copies(tmp_path):
    from scripts.download_dataset import verify_category

    generate_mock_category(str(tmp_path), "grid", num_train=5, num_test=4)
    problems = verify_category(tmp_path / "grid", "grid")
    assert any("train images" in p for p in problems) and any("official" in p for p in problems)
