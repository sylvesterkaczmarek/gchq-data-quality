# (c) Crown Copyright GCHQ
import json

import numpy as np
import pandas as pd
import pytest

from gchq_data_quality.results.models import DataQualityReport, DataQualityResult
from gchq_data_quality.rules import ValuesMatchExpression


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([], []),
        (["a", "b"], ["a", "b"]),
        ([np.nan], [None]),
        ((pd.NaT, "b"), [None, "b"]),
        (
            {"missing": pd.NA, "values": [1, np.nan]},
            {"missing": None, "values": [1, None]},
        ),
        ({"values": []}, {"values": []}),
        (np.array([1.0, np.nan]), [1.0, None]),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), [[1.0, None], [2.0, 3.0]]),
        (np.array(np.nan), None),
    ],
    ids=[
        "empty-list",
        "list",
        "missing-in-list",
        "tuple",
        "nested-dict",
        "empty-nested-list",
        "array",
        "matrix",
        "scalar-array",
    ],
)
def test_nested_sample_exports(
    basic_data_quality_result: DataQualityResult, value: object, expected: object
) -> None:
    result = basic_data_quality_result
    result.records_failed_sample = [{"value": value}]
    expected_samples = [{"value": expected}]

    assert result.to_dict()["records_failed_sample"] == expected_samples
    assert json.loads(result.to_json())["records_failed_sample"] == expected_samples
    spark_row = result._to_spark_schema_df().iloc[0]
    assert json.loads(spark_row["records_failed_sample"]) == expected_samples


def test_sample_export_does_not_mutate_result(
    basic_data_quality_result: DataQualityResult,
) -> None:
    values = [np.nan, pd.NA, pd.NaT]
    sample = {"scalar": np.nan, "nested": {"values": values}}
    basic_data_quality_result.records_failed_sample = [sample]

    basic_data_quality_result.to_json()

    assert np.isnan(sample["scalar"])
    assert sample["nested"]["values"] is values
    assert np.isnan(values[0])
    assert values[1] is pd.NA
    assert values[2] is pd.NaT


def test_expression_report_with_list_samples() -> None:
    df = pd.DataFrame({"tags": [["a", "b"]]})
    result = ValuesMatchExpression(
        field="tags", expression="`tags`.str.len() <= 1"
    ).evaluate(df)
    report = DataQualityReport(results=[result])

    assert result.pass_rate == 0.0
    exported = json.loads(report.to_json())
    assert exported["results"][0]["records_failed_sample"] == [{"tags": ["a", "b"]}]
