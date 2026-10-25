"""Data-invariant and light model checks; no optimisation packages required."""
import numpy as np
import pytest
from src import repro as r


def test_stored_table_and_class_counts():
    df=r.load_data()
    assert df.shape==(24000,14)
    assert df.group.nunique()==800
    assert df.performance.value_counts().to_dict()=={1:19030,0:4970}
    assert np.array_equal(r.label_by_coefficient(df,1.5),df.performance.to_numpy())


def test_threshold_counts():
    df=r.load_data()
    expected={1.25:8396,1.5:4970,1.75:2367,2.0:855}
    assert {coefficient:int((r.label_by_coefficient(df,coefficient)==0).sum())
            for coefficient in expected}==expected


def test_splits():
    df=r.load_data()
    tr,te=r.partitions(df,'job',seed=42)
    assert (len(tr),len(te))==(19200,4800)
    assert len(set(df.group.iloc[tr]) & set(df.group.iloc[te]))==0
    assert len(set(df.group.iloc[tr]))==640
    assert len(set(df.group.iloc[te]))==160
    other_tr,other_te=r.partitions(df,'profile',seed=42)
    assert len(set(df.group.iloc[other_tr]) & set(df.group.iloc[other_te]))==798


def test_softmax_normalisation_and_uniform_weight():
    assert np.allclose(r.softmax(np.zeros(8)),np.ones(8)/8)
    assert np.isclose(r.softmax(np.arange(8)).sum(),1)


def test_scoring_and_smoke_baseline():
    df=r.select_mode(r.load_data(),'smoke')
    tr,te=r.partitions(df,'job',42)
    fit=r.fit_test('XGBoost',df.loc[tr,r.FEATURES],df.performance.iloc[tr],
                   df.loc[te,r.FEATURES],df.performance.iloc[te],11,scaled=True)
    assert 0<=fit['metrics']['macro_f1']<=1
    assert fit['predictions'].shape==(len(te),)
