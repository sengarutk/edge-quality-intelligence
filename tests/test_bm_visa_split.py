"""The VisA adapter in scripts/build_score_bank.py follows the official one-class split."""

from pathlib import Path


def test_visa_split_reads_official_csv(tmp_path: Path):
    from scripts.build_score_bank import VisaSplit

    (tmp_path / "split_csv").mkdir()
    (tmp_path / "split_csv" / "1cls.csv").write_text(
        "object,split,label,image,mask\n"
        "pcb1,train,normal,pcb1/Data/Images/Normal/0001.JPG,\n"
        "pcb1,train,normal,pcb1/Data/Images/Normal/0000.JPG,\n"
        "pcb1,test,normal,pcb1/Data/Images/Normal/0100.JPG,\n"
        "pcb1,test,anomaly,pcb1/Data/Images/Anomaly/0000.JPG,pcb1/Data/Masks/Anomaly/0000.png\n"
        "pcb2,train,normal,pcb2/Data/Images/Normal/0000.JPG,\n", encoding="utf-8")
    train, test = VisaSplit(tmp_path, "pcb1", "train"), VisaSplit(tmp_path, "pcb1", "test")
    assert len(train) == 2 and train.paths[0].endswith("Normal/0000.JPG")
    assert [s[1] for s in test.samples] == [1, 0]  # sorted by path: Anomaly/ before Normal/
    assert test.samples[0][3]["defect_type"] == "anomaly"
