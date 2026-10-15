import unittest
from unittest.mock import patch
from common.data.database import Database, DatabaseError


class Connection:
    read_only = False

    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, query, params=()):
        if not self.read_only:
            raise AssertionError("Connection must be read-only before first query")
        self.queries.append((query, params))
        return self

    def fetchmany(self, count):
        return self.rows[:count]


class DatabaseTests(unittest.TestCase):
    def test_readonly_and_parameters(self):
        conn = Connection([{"value": 1}])
        with patch("psycopg.connect", return_value=conn):
            result = Database("secret").fetch_one("SELECT %s AS value", (1,))
        self.assertEqual(result, {"value": 1})
        self.assertTrue(conn.read_only)
        self.assertEqual(conn.queries[-1], ("SELECT %s AS value", (1,)))
        self.assertIn("statement_timeout", conn.queries[0][0])

    def test_overflow_and_duplicate_are_not_silent(self):
        with patch("psycopg.connect", return_value=Connection([{}, {}])):
            with self.assertRaises(DatabaseError):
                Database("secret", max_rows=1).fetch_all("SELECT 1")
        with patch("psycopg.connect", return_value=Connection([{}, {}])):
            with self.assertRaises(DatabaseError):
                Database("secret").fetch_one("SELECT 1")

    def test_nonquery_rejected_before_connect(self):
        with patch("psycopg.connect") as connect:
            with self.assertRaises(ValueError):
                Database("secret").fetch_all("DELETE FROM any_table")
            connect.assert_not_called()

    def test_driver_error_redacted(self):
        import psycopg
        with patch("psycopg.connect", side_effect=psycopg.OperationalError("password=secret")):
            with self.assertRaises(DatabaseError) as caught:
                Database("secret").fetch_all("SELECT 1")
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
