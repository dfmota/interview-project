import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta


def generate_transaction_fxrates_random_dataframes(
        n_transactions: int = 2000,
        n_fx_days: int = 30,
        currencies: list = None,
        statuses: list = None,
        seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate random 'transactions' and 'fx_rates' DataFrames.
    
    Parameters (all optional):
        n_transactions : int  -> number of transaction rows
        n_fx_days      : int  -> number of FX days to generate
        currencies     : list -> list of currency codes (default: ["EUR", "USD", "GBP"])
        statuses       : list -> list of transaction statuses
        seed           : int  -> random seed for reproducibility

    Returns:
        transactions (DataFrame)
        fx_rates (DataFrame)
    """

    # -----------------------
    # Defaults
    # -----------------------
    if currencies is None:
        currencies = ["EUR", "USD", "GBP"]
    if statuses is None:
        statuses = ["completed", "pending", "failed", "reversed"]

    np.random.seed(seed)

    # -----------------------
    # Generate transactions
    # -----------------------
    transaction_ids = [f"tx_{i:05d}" for i in range(n_transactions)]
    user_ids = np.random.randint(1, 500, size=n_transactions)

    base_datetime = datetime(2024, 1, 1, 12, 0, 0)
    timestamps = [
        base_datetime - timedelta(
            days=int(i % 60),
            hours=int(i % 24),
            minutes=int(i % 60)
        )
        for i in range(n_transactions)
    ]

    amounts = np.round(np.random.uniform(1, 500, size=n_transactions), 2)
    currency_choices = np.random.choice(currencies, size=n_transactions)
    status_choices = np.random.choice(statuses, size=n_transactions, p=None)

    transactions = pd.DataFrame({
        "transaction_id": transaction_ids,
        "user_id": user_ids,
        "timestamp": timestamps,
        "amount": amounts,
        "currency": currency_choices,
        "status": status_choices
    })

    # -----------------------
    # Generate FX rates
    # -----------------------
    start_date = datetime(2024, 1, 1).date()

    fx_data = []
    for day in range(n_fx_days):
        date = start_date + timedelta(days=day)
        for curr in currencies:
            if curr == "EUR":
                rate = 1.0
            elif curr == "USD":
                rate = round(np.random.normal(0.92, 0.02), 4)
            elif curr == "GBP":
                rate = round(np.random.normal(1.15, 0.03), 4)
            else:
                # fallback for unknown currencies
                rate = round(np.random.uniform(0.5, 1.5), 4)

            fx_data.append([date, curr, rate])

    fx_rates = pd.DataFrame(fx_data, columns=["date", "currency", "rate_to_eur"])

    return transactions, fx_rates


def generate_kyc_documents_dataframe(
        n_docs: int = 500,
        doc_types: list = None,
        statuses: list = None,
        seed: int = 123
) -> pd.DataFrame:
    """
    Generate a deterministic KYC documents DataFrame.

    Columns:
        - user_id (int)
        - doc_id (str)
        - doc_type (str)
        - submitted_at (datetime)
        - status (str)
    """

    if doc_types is None:
        doc_types = ["passport", "id_card", "driver_license"]

    if statuses is None:
        statuses = ["approved", "rejected", "pending", "expired"]

    np.random.seed(seed)

    user_ids = np.random.randint(1, 300, size=n_docs)
    doc_ids = [f"doc_{i:05d}" for i in range(n_docs)]
    doc_type_choices = np.random.choice(doc_types, size=n_docs)
    status_choices = np.random.choice(statuses, size=n_docs)

    base_datetime = datetime(2024, 1, 1, 8, 0, 0)
    submitted_at_list = [
        base_datetime + timedelta(
            days=int(i % 30),
            hours=int(i % 12),
            minutes=int(i % 50)
        )
        for i in range(n_docs)
    ]

    df = pd.DataFrame({
        "user_id": user_ids,
        "doc_id": doc_ids,
        "doc_type": doc_type_choices,
        "submitted_at": submitted_at_list,
        "status": status_choices
    })

    return df


def compute_user_verification_status(kyc_documents: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de un DataFrame de documentos KYC, calcula un estado final de verificación por usuario.

    Parámetros
    ----------
    kyc_documents : pd.DataFrame
        Debe contener al menos las columnas:
          - 'user_id' (int)
          - 'doc_id' (str)
          - 'doc_type' (str)
          - 'submitted_at' (datetime o convertible con pd.to_datetime)
          - 'status' (str) – "approved", "rejected", "pending", "expired"

    Reglas de negocio para derivar verification_status por usuario:
      1. Si tiene algún documento "approved" y ningún "rejected" posterior a ese "approved" → "verified".
      2. Si el último documento en el tiempo es "rejected" → "rejected".
      3. Si tiene al menos un "pending" y ninguno "approved" → "pending".
      4. En cualquier otro caso → "unverified".

    Devuelve
    --------
    DataFrame con columnas:
      - user_id
      - verification_status
      - last_update_at (la fecha/hora del último documento del usuario)
    """

    if kyc_documents.empty:
        return pd.DataFrame(columns=["user_id", "verification_status", "last_update_at"])

    df = kyc_documents.copy()

    # Aseguramos tipo datetime
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])

    # Orden global (no es estrictamente necesario, pero ayuda a la legibilidad)
    df = df.sort_values(["user_id", "submitted_at"]).reset_index(drop=True)

    def _compute_for_user(group: pd.DataFrame) -> pd.Series:
        # El grupo ya viene ordenado por submitted_at
        group = group.sort_values("submitted_at")

        last_row = group.iloc[-1]
        last_update_at = last_row["submitted_at"]
        last_status = last_row["status"]

        statuses = group["status"]

        has_approved = (statuses == "approved").any()
        has_pending = (statuses == "pending").any()

        # --- Regla 1: "verified" ---
        if has_approved:
            approved_times = group.loc[group["status"] == "approved", "submitted_at"]
            rejected_times = group.loc[group["status"] == "rejected", "submitted_at"]

            max_approved_time = approved_times.max()
            if rejected_times.empty:
                verification_status = "verified"
            else:
                max_rejected_time = rejected_times.max()
                if max_approved_time >= max_rejected_time:
                    verification_status = "verified"
                else:
                    verification_status = None
        else:
            verification_status = None

        # --- Regla 2: último doc es "rejected" ---
        if verification_status is None:
            if last_status == "rejected":
                verification_status = "rejected"

        # --- Regla 3: tiene "pending" y ningún "approved" ---
        if verification_status is None:
            if has_pending and not has_approved:
                verification_status = "pending"

        # --- Regla 4: resto de casos ---
        if verification_status is None:
            verification_status = "unverified"

        return pd.Series({
            "user_id": group["user_id"].iloc[0],
            "verification_status": verification_status,
            "last_update_at": last_update_at,
        })

    result = (
        df
        .groupby("user_id", group_keys=False)
        .apply(_compute_for_user)
        .reset_index(drop=True)
    )

    return result


def compute_user_volume_eur(transactions: pd.DataFrame,
                            fx_rates: pd.DataFrame,
                            top_n: int | None = None) -> pd.DataFrame:
    """
    Calcula el volumen total en EUR por usuario a partir de:
      - transactions: DataFrame con al menos columnas
          ['timestamp', 'status', 'user_id', 'amount', 'currency']
      - fx_rates: DataFrame con al menos columnas
          ['date', 'currency', 'rate_to_eur']
          
    Pasos:
      1. Filtra solo las transacciones 'completed'.
      2. Convierte cada transacción a EUR usando el FX del día:
         - date = transactions["timestamp"].dt.date
         - merge con fx_rates por ['date', 'currency']
         - Si currency == "EUR" y no hay fila en fx_rates, asume rate_to_eur = 1.0.
      3. Calcula el volumen total en EUR por usuario.
      4. Devuelve DataFrame con columnas:
         ['user_id', 'total_volume_eur'], ordenado desc por volumen.
      5. Si top_n no es None, devuelve solo los top N usuarios.
    """

    # Copias defensivas para no modificar los dataframes originales
    tx = transactions.copy()
    fx = fx_rates.copy()

    # Aseguramos que timestamp es datetime
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    # 1. Filtra solo las transacciones completed
    tx = tx[tx["status"] == "completed"].copy()

    if tx.empty:
        # No hay transacciones completadas: devolvemos DF vacío con las columnas esperadas
        return pd.DataFrame(columns=["user_id", "total_volume_eur"])

    # 2. Preparamos la columna date en ambos dataframes (tipo date, no datetime)
    tx["date"] = tx["timestamp"].dt.date
    fx["date"] = pd.to_datetime(fx["date"]).dt.date

    # Merge con fx_rates por date y currency
    merged = tx.merge(
        fx[["date", "currency", "rate_to_eur"]],
        on=["date", "currency"],
        how="left"
    )

    # Si currency == "EUR" y no hay fila en fx_rates,
    # asumimos rate_to_eur = 1.0
    eur_mask = (merged["currency"] == "EUR") & (merged["rate_to_eur"].isna())
    merged.loc[eur_mask, "rate_to_eur"] = 1.0

    # Si todavía hay NaN en rate_to_eur (monedas sin tipo de cambio),
    # las descartamos o podrías lanzar error
    merged = merged.dropna(subset=["rate_to_eur"])

    # Convertimos el importe a EUR
    merged["amount_eur"] = merged["amount"] * merged["rate_to_eur"]

    # 3. Calcula el volumen total en EUR por usuario
    result = (
        merged
        .groupby("user_id", as_index=False)["amount_eur"]
        .sum()
        .rename(columns={"amount_eur": "total_volume_eur"})
    )

    # 4. Ordenado desc por volumen
    result = result.sort_values("total_volume_eur", ascending=False)

    # 5. Opcional: top N usuarios
    if top_n is not None:
        result = result.head(top_n)
    
    return result


# --- Suspicious users detection ---
def detect_suspicious_users(transactions: pd.DataFrame,
                            min_failures: int = 5,
                            window_minutes: int = 10) -> pd.DataFrame:
    """
    Detecta usuarios sospechosos según:
      - Un usuario es sospechoso si tiene al menos `min_failures` transacciones
        con status "failed" en cualquier ventana de `window_minutes` minutos.

    Parámetros
    ----------
    transactions : pd.DataFrame
        Debe contener al menos:
          - 'user_id'
          - 'timestamp' (datetime o convertible con pd.to_datetime)
          - 'status'
    min_failures : int
        Número mínimo de fallos en la ventana para considerar sospechoso.
    window_minutes : int
        Tamaño de la ventana en minutos.

    Devuelve
    --------
    pd.DataFrame con columnas:
        - user_id
        - first_window_start
        - first_window_end
        - num_failed_in_window
    Una fila por usuario sospechoso.
    """

    # 1. Nos quedamos solo con transacciones failed
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["status"] == "failed"].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["user_id", "first_window_start", "first_window_end", "num_failed_in_window"]
        )

    # 2. Ordenamos por user_id y timestamp
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    results = []
    window_delta = pd.Timedelta(minutes=window_minutes)

    # 3. Sliding window por usuario
    for user_id, group in df.groupby("user_id", sort=False):
        timestamps = group["timestamp"].values
        n = len(timestamps)
        left = 0

        for right in range(n):
            while timestamps[right] - timestamps[left] > window_delta:
                left += 1

            window_size = right - left + 1

            if window_size >= min_failures:
                results.append({
                    "user_id": user_id,
                    "first_window_start": timestamps[left],
                    "first_window_end": timestamps[right],
                    "num_failed_in_window": window_size
                })
                break

    if not results:
        return pd.DataFrame(
            columns=["user_id", "first_window_start", "first_window_end", "num_failed_in_window"]
        )

    return pd.DataFrame(results)

def generate_random_checks_dataframe(
    n_rows: int = 1000,
    decision_threshold: float = 0.5,
    start_date: datetime = None,
    end_date: datetime = None,
) -> pd.DataFrame:
    """
    Generate a random dataframe simulating identity checks with the fields:
    check_id, user_id, check_type, created_at, model_score, ground_truth, decision.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    decision_threshold : float
        Threshold used to simulate the 'decision' column (accepted vs rejected).
    start_date : datetime
        Start of the date range for created_at.
    end_date : datetime
        End of the date range for created_at.

    Returns
    -------
    df : pd.DataFrame
        Randomly generated dataframe.
    """

    # Set default date window if not provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()

    # Random timestamps
    created_at = pd.to_datetime(
        np.random.randint(
            int(start_date.timestamp()),
            int(end_date.timestamp()),
            size=n_rows
        ),
        unit="s"
    )

    # Random ground truth distribution
    ground_truth_values = np.random.choice(
        ["genuine", "fraud"], 
        size=n_rows, 
        p=[0.85, 0.15]  # typical imbalance
    )

    # Model score: genuine should have slightly higher scores
    model_score = np.where(
        ground_truth_values == "genuine",
        np.random.normal(loc=0.75, scale=0.15, size=n_rows),
        np.random.normal(loc=0.30, scale=0.15, size=n_rows)
    )
    model_score = np.clip(model_score, 0, 1)

    # Simulate decisions using a threshold (legacy rules)
    decision = np.where(model_score >= decision_threshold, "accepted", "rejected")

    # Create DataFrame
    df = pd.DataFrame({
        "check_id": [str(uuid.uuid4()) for _ in range(n_rows)],
        "user_id": np.random.randint(1, 5000, size=n_rows),
        "check_type": np.random.choice(["id_document", "selfie"], size=n_rows),
        "created_at": created_at,
        "model_score": model_score,
        "ground_truth": ground_truth_values,
        "decision": decision
    })

    return df
