import pytest
import pandas as pd

@pytest.fixture(scope="session")
def shared_dataframes():
    from src.data import generate_transaction_fxrates_random_dataframes
    transactions_df, fx_rates_df = generate_transaction_fxrates_random_dataframes()
    yield transactions_df, fx_rates_df

    # Cleanup after all tests have executed
    del transactions_df
    del fx_rates_df

def test_revolut_user_volumen(shared_dataframes: pd.DataFrame):
    """Example test to be replaced with real assertions.

    This test is here just to verify that your test setup works.
    Replace it with real tests for your Revolut solution.
    """

    # Import the generator function
    from src.data import compute_user_volume_eur

    # Generate the DataFrames used for testing
    transactions_df, fx_rates_df = shared_dataframes

    # Example assertions (replace with real test logic)
    assert transactions_df is not None
    assert fx_rates_df is not None

    df_top10 = compute_user_volume_eur(transactions_df, fx_rates_df, 10)

    expected = [318.0, 1172.525036]
    actual = df_top10.iloc[0][["user_id", "total_volume_eur"]].tolist()

    assert expected==actual


def test_detect_suspicious_users(shared_dataframes: pd.DataFrame):
    """Test suspicious user detection using randomly generated data."""

    from src.data import detect_suspicious_users

    # Generate random transactions and fx rates
    transactions_df, _ = shared_dataframes

    # Call detection function
    suspicious_df = detect_suspicious_users(transactions_df)

    # Basic assertions to verify output structure
    assert suspicious_df is not None
    assert "user_id" in suspicious_df.columns

    # Ensure results are consistent with input types
    # (depending on the random seed, suspicious_df may be empty or not)
    assert isinstance(suspicious_df, type(transactions_df))


def test_user_verification_status():
    """Test user verification status using randomly generated data."""

    from src.data import generate_kyc_documents_dataframe
    from src.data import compute_user_verification_status

    # Generate random transactions and fx rates
    kyc_docs_df = generate_kyc_documents_dataframe()

    # Call detection function
    status_df = compute_user_verification_status(kyc_docs_df)

    # Basic assertions to verify output structure
    assert status_df is not None

    counts = status_df["verification_status"].value_counts()

    num_pending = counts.get("pending", 0)
    num_verified = counts.get("verified", 0)
    num_rejected = counts.get("rejected", 0)
    num_unverified = counts.get("unverified", 0)

    assert int(num_pending)==50
    assert int(num_verified)==85
    assert int(num_rejected)==62
    assert int(num_unverified)==35
