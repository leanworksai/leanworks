import datetime

from leanworks.agent.tools.bigquery import BigQueryTool


class MockField:
    def __init__(self, name, field_type, description=""):
        self.name = name
        self.field_type = field_type
        self.description = description


class MockTable:
    def __init__(self, table_id, schema=None, description=""):
        self.table_id = table_id
        self.schema = schema or []
        self.description = description


class MockTableItem:
    def __init__(self, table_id):
        self.table_id = table_id
        # The tool only passes this value back into get_table, so a simple string works
        self.reference = table_id


class MockQueryJob:
    def __init__(self, sql, rows):
        self.sql = sql
        self._rows = rows
        self.job_id = "job_123"

    def result(self):
        return self._rows


class MockBigQueryClient:
    def __init__(self, tables_by_id=None, should_raise=False):
        self.project = "test-project"
        self._tables_by_id = tables_by_id or {}
        self.should_raise = should_raise
        self.last_query = None

    def list_tables(self, dataset_ref):
        return [MockTableItem(tid) for tid in self._tables_by_id.keys()]

    def get_table(self, reference):
        return self._tables_by_id[reference]

    def query(self, sql):
        if self.should_raise:
            raise RuntimeError("simulated query failure")
        self.last_query = sql
        rows = [
            {
                "orders": 42,
                "date_col": datetime.date(2025, 1, 2),
                "dt_col": datetime.datetime(2025, 1, 2, 3, 4, 5),
            }
        ]
        return MockQueryJob(sql, rows)


class MockBQWrapper:
    def __init__(self, bq_client, client_name):
        self.bq_client = bq_client
        self.client_name = client_name


def build_tool_with_schema(client_name="test_client"):
    # Define one table with a unix timestamp column stored as FLOAT and described accordingly
    orders_table = MockTable(
        table_id="orders",
        schema=[
            MockField("id", "INT64", ""),
            MockField("created_at", "FLOAT64", "Unix timestamp in millis"),
        ],
        description="Orders table",
    )
    mock_client = MockBigQueryClient(tables_by_id={"orders": orders_table})
    wrapper = MockBQWrapper(mock_client, client_name)
    return BigQueryTool(wrapper), mock_client


def test_query_bigquery_compiles_and_executes_with_unix_ts_handling():
    tool, mock_client = build_tool_with_schema()

    spec = {
        "from": [{"table": "orders", "alias": "o"}],
        "select": [{"expr": "count(*)", "as": "orders"}],
        "where": [
            {"column": "o.created_at", "op": ">=", "value": "2025-01-01T00:00:00Z"}
        ],
        "order_by": [{"expr": "orders", "dir": "DESC"}],
        "limit": 10,
    }

    results = tool.query_bigquery(spec)

    # Verify SQL was generated and executed
    assert mock_client.last_query is not None
    sql = mock_client.last_query

    # Fully-qualified and backticked table name
    assert "FROM `leanworks.test_client.orders` AS o" in sql

    # Verify Unix millis conversion and CAST to FLOAT64 for comparison
    assert "o.created_at >= CAST(UNIX_MILLIS(TIMESTAMP('2025-01-01T00:00:00Z')) AS FLOAT64)" in sql

    # Verify results are formatted with ISO strings for date/datetime
    assert isinstance(results, list) and len(results) == 1
    row = results[0]
    assert row["orders"] == 42
    assert row["date_col"] == "2025-01-02"
    assert row["dt_col"].startswith("2025-01-02T03:04:05")


def test_query_bigquery_error_handling_returns_error_object():
    # Build tool but replace client with one that raises on query
    tool, _ = build_tool_with_schema()
    failing_client = MockBigQueryClient(should_raise=True)
    tool.bq_client_wrapper.bq_client = failing_client

    spec = {
        "from": [{"table": "orders"}],
        "select": ["count(*)"],
        "limit": 1,
    }

    result = tool.query_bigquery(spec)
    assert isinstance(result, dict)
    assert "error" in result


if __name__ == "__main__":
    test_query_bigquery_compiles_and_executes_with_unix_ts_handling()
    test_query_bigquery_error_handling_returns_error_object()
    print("BigQueryTool tests passed")


