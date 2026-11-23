"""Failed model calls keep scrubbed diagnostics while the raised error stays generic."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import socket
import threading
import time
import unittest

from policyweave.adapters.compatible import Connection

SECRET = "synthetic-secret-body"


def serve(status, body, delay=0):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(delay)
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:
                pass  # the timeout test's client has already gone

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class FailureDiagnosticTests(unittest.TestCase):
    def connection(self, status, body, delay=0, timeout=5, protocol="chat-completions"):
        server = serve(status, body, delay)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return self.open(server.server_port, timeout, protocol)

    def open(self, port, timeout=5, protocol="chat-completions"):
        return Connection(dict(
            protocol=protocol, baseUrl="http://127.0.0.1:%d/v1" % port,
            modelId="fixture", maxTokens=100, timeoutSeconds=timeout, apiKey="synthetic-key"))

    def test_http_status_is_kept_without_the_body(self):
        connection = self.connection(429, dict(error=SECRET))
        with self.assertRaisesRegex(RuntimeError, "^Model connection failed"):
            connection("plan", "data")
        self.assertEqual(connection.failures[0]["httpStatus"], 429)
        self.assertNotIn(SECRET, repr(connection.failures))
        self.assertNotIn("synthetic-key", repr(connection.failures))

    def test_incomplete_finish_reason_is_kept(self):
        connection = self.connection(200, dict(choices=[dict(finish_reason="length", message=dict(content=SECRET))]))
        with self.assertRaises(RuntimeError):
            connection("plan", "data")
        failure = connection.failures[0]
        self.assertEqual((failure["finishReason"], failure["errorType"]), ("length", "ValueError"))
        self.assertNotIn(SECRET, repr(failure))

    def test_read_timeout_is_recorded_and_raised_unchanged(self):
        connection = self.connection(200, dict(choices=[]), delay=2, timeout=1)
        with self.assertRaises(TimeoutError):
            connection("plan", "data")
        failure = connection.failures[0]
        self.assertEqual((failure["errorType"], failure["timeoutSeconds"]), ("TimeoutError", 1))
        self.assertGreaterEqual(failure["elapsedSeconds"], 0.9)

    def test_refused_connection_keeps_only_the_reason_class(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        connection = self.open(port)
        with self.assertRaises(RuntimeError):
            connection("plan", "data")
        failure = connection.failures[0]
        self.assertEqual((failure["errorType"], failure["reason"]), ("URLError", "ConnectionRefusedError"))
        self.assertNotIn("127.0.0.1", repr(failure))

    def test_invalid_json_keeps_size_but_not_text(self):
        connection = self.connection(200, SECRET.encode())
        with self.assertRaises(RuntimeError):
            connection("plan", "data")
        failure = connection.failures[0]
        self.assertEqual((failure["errorType"], failure["responseBytes"]), ("JSONDecodeError", len(SECRET)))
        self.assertNotIn("detail", failure)
        self.assertNotIn(SECRET, repr(failure))

    def test_messages_stop_reason_is_kept(self):
        body = dict(type="message", stop_reason="max_tokens", content=[dict(type="text", text=SECRET)])
        connection = self.connection(200, body, protocol="messages")
        with self.assertRaises(RuntimeError):
            connection("plan", "data")
        failure = connection.failures[0]
        self.assertEqual(failure["finishReason"], "max_tokens")
        self.assertEqual(failure["detail"], "Model response is incomplete or refused.")
        self.assertNotIn(SECRET, repr(failure))

    def test_success_records_no_failure(self):
        connection = self.connection(200, dict(choices=[dict(finish_reason="stop", message=dict(content="{}"))]))
        self.assertEqual(connection("plan", "data"), "{}")
        self.assertEqual(connection.failures, [])


if __name__ == "__main__":
    unittest.main()
