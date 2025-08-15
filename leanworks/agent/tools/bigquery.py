from google.cloud import bigquery
import json
import logging
import datetime
import re

logger = logging.getLogger(__name__)


class BigQueryTool:
    def __init__(self, bq_client_wrapper):
        """
        Initialize BigQueryTool with a BigQuery client wrapper.

        On init, load all table schemas for dataset `leanworks.{client_name}` so
        the agent can see available tables and columns when deciding what to query.

        Args:
            bq_client_wrapper: An object with attributes `bq_client` (google.cloud.bigquery.Client)
                               and `client_name` (dataset suffix under project `leanworks`).
        """
        self.bq_client_wrapper = bq_client_wrapper

        # Load all table schemas into a list on initialization using the attached script pattern
        # Schema structure: [ { 'bq_table_path': 'leanworks.{client}.{table}', 'schema': [field_dict, ...] }, ... ]
        self.table_schemas = []
        try:
            dataset_ref = bigquery.DatasetReference("leanworks", self.bq_client_wrapper.client_name)
            tables = self.bq_client_wrapper.bq_client.list_tables(dataset_ref)
            for table_item in tables:
                try:
                    table = self.bq_client_wrapper.bq_client.get_table(table_item.reference)
                    # Prefer API repr so the schema is always JSON serializable
                    try:
                        schema_list = [getattr(f, "to_api_repr")() for f in (table.schema or [])]
                    except Exception:
                        # Fallback to string if to_api_repr is not available
                        schema_list = [{
                            "name": str(getattr(f, "name", "")),
                            "type": str(getattr(f, "field_type", getattr(f, "type", ""))),
                            "description": str(getattr(f, "description", ""))
                        } for f in (table.schema or [])]
                    self.table_schemas.append({
                        "bq_table_path": f"leanworks.{self.bq_client_wrapper.client_name}.{table.table_id}",
                        "schema": schema_list,
                        "description": getattr(table, "description", "") or ""
                    })
                except Exception as e:
                    logger.warning(f"Failed to load schema for table {table_item.table_id}: {str(e)}")
        except Exception as e:
            logger.warning(
                f"Failed to list tables for dataset leanworks.{getattr(self.bq_client_wrapper, 'client_name', 'unknown')}: {str(e)}"
            )
        catalog_lines = []
        for entry in self.table_schemas:
            path = entry.get("bq_table_path", "")
            if not path:
                continue
            cols = []
            for f in entry.get("schema", []):
                try:
                    name = f.get("name", "")
                    ftype = f.get("type", "")
                    cols.append(f"{name} {ftype}".strip())
                except Exception:
                    cols.append(str(f))
            catalog_lines.append(f"- {path}: {', '.join(cols)}")
        self.table_catalog_brief = "\n".join(catalog_lines)
        self.schemas_json = json.dumps(self.table_schemas, ensure_ascii=False)
        # Build a merged tables-and-schemas view in one list, including column descriptions if present
        merged_lines = []
        for entry in self.table_schemas:
            path = entry.get("bq_table_path", "")
            if not path:
                continue
            cols = []
            for f in entry.get("schema", []):
                try:
                    name = f.get("name", "")
                    ftype = f.get("type", "")
                    desc = f.get("description") or ""
                    if desc:
                        cols.append(f"{name} {ftype} - {desc}".strip())
                    else:
                        cols.append(f"{name} {ftype}".strip())
                except Exception:
                    cols.append(str(f))
            merged_lines.append(f"- {path}: {', '.join(cols)}")
        self.tables_and_schemas = "\n".join(merged_lines)

        # Build a set of column names that are Unix timestamps (by description), and detect units
        # Map: column_name -> "seconds" | "millis"
        self.unix_ts_columns = {}
        for entry in self.table_schemas:
            for f in entry.get("schema", []):
                try:
                    name = (f.get("name") or "").strip()
                    desc = (f.get("description") or "").lower()
                    if not name:
                        continue
                    if "unix" in desc and "timestamp" in desc:
                        unit = "millis" if ("millis" in desc or "ms" in desc) else "seconds"
                        # Preserve the most specific (millis overrides seconds if seen later)
                        prev = self.unix_ts_columns.get(name)
                        if prev != "millis":
                            self.unix_ts_columns[name] = unit
                except Exception:
                    continue

    def _fully_qualify_table(self, table_name: str) -> str:
        # Backtick-quote and prefix dataset if needed
        if not table_name:
            return table_name
        if table_name.startswith("`") and table_name.endswith("`"):
            return table_name
        # Already qualified as project.dataset.table or dataset.table
        if table_name.count(".") >= 1:
            return f"`{table_name}`"
        dataset = f"leanworks.{getattr(self.bq_client_wrapper, 'client_name', 'unknown')}"
        return f"`{dataset}.{table_name}`"

    def _format_literal(self, value):
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, (int, float)):
            return str(value)
        # Strings: if already function-like, pass through
        s = str(value)
        if re.match(r"^(TIMESTAMP|DATE|DATETIME)\(.*\)$", s, flags=re.IGNORECASE):
            return s
        # Quote single quotes inside
        s_escaped = s.replace("'", "\\'")
        return f"'{s_escaped}'"

    def _column_base_name(self, col: str) -> str:
        # Strip alias prefix like "o.created_at" -> "created_at"
        if not isinstance(col, str):
            return col
        return col.split(".")[-1]

    def _compile_where_clause(self, where_list):
        if not where_list:
            return ""
        parts = []
        for cond in where_list:
            if not isinstance(cond, dict):
                continue
            column = cond.get("column")
            op = (cond.get("op") or "=").upper()
            value = cond.get("value")
            if not column:
                continue
            base = self._column_base_name(column)
            unit = self.unix_ts_columns.get(base)
            # Helper: format value for unix timestamp columns saved as FLOAT
            def _format_unix_value(v, unit: str) -> str:
                # If ISO string, convert to UNIX_* and cast to FLOAT64 to match FLOAT columns
                if isinstance(v, str):
                    inner = self._format_literal(v)
                    if unit == "millis":
                        return f"CAST(UNIX_MILLIS(TIMESTAMP({inner})) AS FLOAT64)"
                    return f"CAST(UNIX_SECONDS(TIMESTAMP({inner})) AS FLOAT64)"
                # If numeric provided, assume already in the right unit; cast to FLOAT64
                if isinstance(v, (int, float)):
                    return f"CAST({v} AS FLOAT64)"
                # Fallback
                return self._format_literal(v)

            # BETWEEN supports list [low, high]
            if op == "BETWEEN" and isinstance(value, (list, tuple)) and len(value) == 2:
                low, high = value[0], value[1]
                if unit in ("millis", "seconds"):
                    expr = f"{column} BETWEEN {_format_unix_value(low, unit)} AND {_format_unix_value(high, unit)}"
                else:
                    expr = f"{column} BETWEEN {self._format_literal(low)} AND {self._format_literal(high)}"
                parts.append(expr)
                continue
            # IN and NOT IN
            if op in ("IN", "NOT IN") and isinstance(value, (list, tuple)):
                if unit in ("millis", "seconds"):
                    list_sql = ", ".join([_format_unix_value(v, unit) for v in value])
                else:
                    list_sql = ", ".join([self._format_literal(v) for v in value])
                parts.append(f"{column} {op} ({list_sql})")
                continue
            # Simple comparisons
            if unit in ("millis", "seconds"):
                parts.append(f"{column} {op} {_format_unix_value(value, unit)}")
            else:
                parts.append(f"{column} {op} {self._format_literal(value)}")
        return (" WHERE " + " AND ".join(parts)) if parts else ""

    def _compile_query_spec(self, spec: dict) -> str:
        if not isinstance(spec, dict):
            raise ValueError("spec must be an object")
        from_items = spec.get("from", []) or []
        if not from_items:
            raise ValueError("spec.from must contain at least one table")
        # FROM and JOINs
        main = from_items[0]
        main_table = self._fully_qualify_table(main.get("table"))
        main_alias = main.get("alias")
        from_sql = f"FROM {main_table}"
        if main_alias:
            from_sql += f" AS {main_alias}"

        join_sql_parts = []
        for j in spec.get("joins", []) or []:
            if not isinstance(j, dict):
                continue
            jtype = (j.get("type") or "INNER").upper()
            jtable = self._fully_qualify_table(j.get("table"))
            jalias = j.get("alias")
            jon = j.get("on")
            if not jtable or not jon:
                continue
            piece = f" {jtype} JOIN {jtable}"
            if jalias:
                piece += f" AS {jalias}"
            piece += f" ON {jon}"
            join_sql_parts.append(piece)

        # SELECT
        select_list = spec.get("select", []) or []
        if not select_list:
            raise ValueError("spec.select must contain at least one expression")
        select_sql_parts = []
        for s in select_list:
            if isinstance(s, dict):
                expr = s.get("expr") or ""
                alias = s.get("as")
                if not expr:
                    continue
                if alias:
                    select_sql_parts.append(f"{expr} AS {alias}")
                else:
                    select_sql_parts.append(expr)
            elif isinstance(s, str):
                select_sql_parts.append(s)
        select_sql = ", ".join(select_sql_parts)

        # WHERE (accept both 'where' and 'filters' synonyms)
        where_sql = self._compile_where_clause(spec.get("where") or spec.get("filters") or [])

        # GROUP BY
        group_by = spec.get("group_by") or []
        group_sql = f" GROUP BY {', '.join(group_by)}" if group_by else ""

        # ORDER BY
        order_by = spec.get("order_by") or []
        if order_by:
            order_parts = []
            for o in order_by:
                if isinstance(o, dict):
                    expr = o.get("expr") or ""
                    direction = (o.get("dir") or o.get("direction") or "ASC").upper()
                    if expr:
                        order_parts.append(f"{expr} {direction}")
                elif isinstance(o, str):
                    order_parts.append(o)
            order_sql = f" ORDER BY {', '.join(order_parts)}" if order_parts else ""
        else:
            order_sql = ""

        # LIMIT
        limit = spec.get("limit")
        limit_sql = f" LIMIT {int(limit)}" if isinstance(limit, int) and limit > 0 else ""

        sql = f"SELECT {select_sql} {from_sql}{''.join(join_sql_parts)}{where_sql}{group_sql}{order_sql}{limit_sql}"
        logger.info("Compiled QuerySpec into SQL: %s", sql)
        return sql

    def _rewrite_sql_for_unix_timestamp(self, sql: str) -> str:
        # If no candidate columns present, return early
        if not self.unix_ts_columns:
            return sql
        original_sql = sql

        def make_replacement(col: str, unit: str, text: str) -> str:
            # Skip if already using UNIX_ functions near this column
            # Basic operator comparisons
            pattern = re.compile(
                rf"(?i)(?<![\w\.])({re.escape(col)})(\s*(=|!=|>=|<=|>|<)\s*)('([^']+)'|TIMESTAMP\('([^']+)'\)|DATETIME\('([^']+)'\))"
            )
            func = "UNIX_MILLIS" if unit == "millis" else "UNIX_SECONDS"

            def repl(m: re.Match) -> str:
                left = m.group(1)
                op = m.group(2)
                literal = m.group(4)
                return f"{left}{op}{func}(TIMESTAMP({literal if literal.startswith('TIMESTAMP(') else literal}))"

            text = pattern.sub(repl, text)

            # BETWEEN pattern with two literals
            pattern_between = re.compile(
                rf"(?i)(?<![\w\.])({re.escape(col)})\s+BETWEEN\s+('([^']+)'|TIMESTAMP\('([^']+)'\)|DATETIME\('([^']+)'\))\s+AND\s+('([^']+)'|TIMESTAMP\('([^']+)'\)|DATETIME\('([^']+)'\))"
            )

            def repl_between(m: re.Match) -> str:
                left = m.group(1)
                lit1 = m.group(2)
                lit2 = m.group(6)
                return (
                    f"{left} BETWEEN {func}(TIMESTAMP({lit1 if lit1.startswith('TIMESTAMP(') else lit1})) "
                    f"AND {func}(TIMESTAMP({lit2 if lit2.startswith('TIMESTAMP(') else lit2}))"
                )

            text = pattern_between.sub(repl_between, text)
            return text

        # Apply replacements for each known column name
        for col, unit in self.unix_ts_columns.items():
            # Only attempt if the column name appears in SQL to reduce overhead
            if re.search(rf"(?i)(?<![\w\.]){re.escape(col)}(?![\w])", sql):
                sql = make_replacement(col, unit, sql)

        if sql != original_sql:
            logger.info("Rewrote SQL for Unix timestamp columns. New SQL: %s", sql)
        return sql
        
    @property
    def query_bigquery_property(self):
        description = f"""
        Run a BigQuery query against `leanworks.{self.bq_client_wrapper.client_name}`.

        This tool is strictly READ-ONLY. It will NEVER delete, update, insert, merge, truncate, create, drop, alter, or otherwise write to any table. It only compiles and executes SELECT queries derived from the provided QuerySpec.

        Provide `spec`: a JSON QuerySpec that the tool compiles into safe SELECT SQL.

        QuerySpec fields (JSON):
        - from: [ {{"table": "orders", "alias": "o"}} ]
        - select: [ {{"expr": "count(*)", "as": "orders"}} ]
        - where: [ {{"column": "o.created_at", "op": ">=", "value": "2025-08-01"}} ]
        - joins: [ {{"type": "LEFT", "table": "users", "alias": "u", "on": "o.user_id = u.id"}} ]
        - group_by: ["o.user_id"]
        - order_by: [ {{"expr": "orders", "dir": "DESC"}} ]
        - limit: 1000

        Notes:
        - Read-only: do not attempt any DML or DDL (INSERT, UPDATE, DELETE, MERGE, TRUNCATE, CREATE, DROP, ALTER). Only SELECT with joins/filters/aggregations/ordering/limits is allowed.
        - All tables are fully qualified to `leanworks.{self.bq_client_wrapper.client_name}`.
        - If a column's description contains 'Unix timestamp' and the column is stored as FLOAT, provide ISO 8601 strings in the QuerySpec (e.g., "2025-08-01T00:00:00Z"). The compiler converts them to UNIX_SECONDS/UNIX_MILLIS and CASTs to FLOAT64 for correct comparisons (including BETWEEN and IN).

        Tables and schemas:
        {self.tables_and_schemas}
        """
        return {
            "type": "custom",
            "name": "query_bigquery",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "object",
                        "description": "QuerySpec to compile into SQL (required).",
                    }
                },
                "required": ["spec"]
            }
        }

    def query_bigquery(self, spec: dict):
        sql = None
        try:
            sql = self._compile_query_spec(spec)
            client_name = getattr(self.bq_client_wrapper, 'client_name', 'unknown')
            project = getattr(getattr(self.bq_client_wrapper, 'bq_client', None), 'project', 'unknown')
            start_time = datetime.datetime.now()

            logger.info(
                f"BigQuery tool call: project={project}, dataset=leanworks.{client_name}, sql={sql}"
            )

            # Rewrite SQL to handle comparisons against Unix timestamp integer columns
            sql = self._rewrite_sql_for_unix_timestamp(sql)

            query_job = self.bq_client_wrapper.bq_client.query(sql)
            results_iter = query_job.result()

            results = []
            for row in results_iter:
                row_dict = dict(row)
                for k, v in list(row_dict.items()):
                    if isinstance(v, (datetime.date, datetime.datetime)):
                        row_dict[k] = v.isoformat()
                results.append(row_dict)
            duration_ms = int((datetime.datetime.now() - start_time).total_seconds() * 1000)
            job_id = getattr(query_job, 'job_id', 'unknown')
            logger.info(
                f"BigQuery tool completed: job_id={job_id}, rows={len(results)}, duration_ms={duration_ms}"
            )
            return results
        except Exception as e:
            client_name = getattr(self.bq_client_wrapper, 'client_name', 'unknown')
            project = getattr(getattr(self.bq_client_wrapper, 'bq_client', None), 'project', 'unknown')
            try:
                sql_snippet = f", sql={sql}" if sql else ""
            except Exception:
                sql_snippet = ""
            logger.error(f"BigQuery tool failed: project={project}, dataset=leanworks.{client_name}, error={str(e)}{sql_snippet}")
            return {"error": f"query_bigquery failed: {str(e)}"}


