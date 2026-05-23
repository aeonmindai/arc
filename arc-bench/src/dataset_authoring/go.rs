//! Go trajectories (5).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        go_017_context_timeout(),
        go_018_errgroup(),
        go_019_grpc_server(),
        go_020_table_test(),
        super::long_trajectories::go_long(),
    ]
}

#[allow(dead_code)]
fn go_016_http_handler() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a GET /v1/healthz handler in cmd/server/main.go that returns 200 OK with {\"status\":\"ok\"} as JSON. Use net/http only.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"cmd/server/main.go"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package main\n\nimport (\n\t\"log\"\n\t\"net/http\"\n)\n\nfunc main() {\n\tmux := http.NewServeMux()\n\tlog.Fatal(http.ListenAndServe(\":8080\", mux))\n}\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_002",
        "write_file",
        json!({
            "path":"cmd/server/main.go",
            "contents":"package main\n\nimport (\n\t\"encoding/json\"\n\t\"log\"\n\t\"net/http\"\n)\n\nfunc healthz(w http.ResponseWriter, r *http.Request) {\n\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tw.WriteHeader(http.StatusOK)\n\t_ = json.NewEncoder(w).Encode(map[string]string{\"status\": \"ok\"})\n}\n\nfunc main() {\n\tmux := http.NewServeMux()\n\tmux.HandleFunc(\"GET /v1/healthz\", healthz)\n\tlog.Fatal(http.ListenAndServe(\":8080\", mux))\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 400 bytes to cmd/server/main.go"));
    turns.push(assistant_text(
        "Handler registered using Go 1.22's method-aware mux pattern.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_016",
        "go",
        src("github.com/golang/go", "BSD-3-Clause", "5b9c89e"),
        turns,
    )
}

fn go_017_context_timeout() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "fetchOrders in client.go takes a context but doesn't propagate a per-call timeout. Add a 3s deadline derived from the caller's context, and return the underlying error wrapped with %w.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"client.go"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package orders\n\nimport (\n\t\"context\"\n\t\"encoding/json\"\n\t\"net/http\"\n)\n\nfunc fetchOrders(ctx context.Context, url string) ([]Order, error) {\n\treq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)\n\tif err != nil { return nil, err }\n\tresp, err := http.DefaultClient.Do(req)\n\tif err != nil { return nil, err }\n\tdefer resp.Body.Close()\n\tvar out []Order\n\treturn out, json.NewDecoder(resp.Body).Decode(&out)\n}\n",
    ));
    turns.push(assistant_call(
        "Adding context.WithTimeout.",
        "call_002",
        "write_file",
        json!({
            "path":"client.go",
            "contents":"package orders\n\nimport (\n\t\"context\"\n\t\"encoding/json\"\n\t\"fmt\"\n\t\"net/http\"\n\t\"time\"\n)\n\nfunc fetchOrders(ctx context.Context, url string) ([]Order, error) {\n\tctx, cancel := context.WithTimeout(ctx, 3*time.Second)\n\tdefer cancel()\n\treq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)\n\tif err != nil { return nil, fmt.Errorf(\"build request: %w\", err) }\n\tresp, err := http.DefaultClient.Do(req)\n\tif err != nil { return nil, fmt.Errorf(\"do request: %w\", err) }\n\tdefer resp.Body.Close()\n\tvar out []Order\n\tif err := json.NewDecoder(resp.Body).Decode(&out); err != nil {\n\t\treturn nil, fmt.Errorf(\"decode response: %w\", err)\n\t}\n\treturn out, nil\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 614 bytes to client.go"));
    turns.push(assistant_text(
        "fetchOrders now derives a 3s timeout from the caller's ctx and wraps every error with a descriptive %w.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_017",
        "go",
        src("github.com/grpc/grpc-go", "Apache-2.0", "f3b8d22"),
        turns,
    )
}

fn go_018_errgroup() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "In ingest/parallel.go we iterate sources sequentially. Use golang.org/x/sync/errgroup to fan out one goroutine per source, return the first error, and cancel siblings via the group's context.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"ingest/parallel.go"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package ingest\n\nimport \"context\"\n\nfunc ingestAll(ctx context.Context, sources []string, fetch func(context.Context, string) error) error {\n\tfor _, s := range sources {\n\t\tif err := fetch(ctx, s); err != nil { return err }\n\t}\n\treturn nil\n}\n",
    ));
    turns.push(assistant_call(
        "Switching to errgroup.WithContext.",
        "call_002",
        "write_file",
        json!({
            "path":"ingest/parallel.go",
            "contents":"package ingest\n\nimport (\n\t\"context\"\n\n\t\"golang.org/x/sync/errgroup\"\n)\n\nfunc ingestAll(ctx context.Context, sources []string, fetch func(context.Context, string) error) error {\n\tg, gctx := errgroup.WithContext(ctx)\n\tfor _, s := range sources {\n\t\ts := s\n\t\tg.Go(func() error { return fetch(gctx, s) })\n\t}\n\treturn g.Wait()\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 348 bytes to ingest/parallel.go"));
    turns.push(assistant_text(
        "Sources now fan out concurrently; the group context cancels siblings as soon as one fetch errors.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_018",
        "go",
        src("github.com/golang/sync", "BSD-3-Clause", "8a2f4b9"),
        turns,
    )
}

fn go_019_grpc_server() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Wire a gRPC server in cmd/grpc/main.go. Register the existing pb.OrdersServer from internal/orders. Use grpc.NewServer with the InsecureCredentials default and listen on :9000.",
    ));
    turns.push(assistant_call(
        "Reading the impl.",
        "call_001",
        "read_file",
        json!({"path":"internal/orders/server.go"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package orders\n\nimport pb \"example.com/proto/orders\"\n\ntype Server struct {\n\tpb.UnimplementedOrdersServer\n}\n",
    ));
    turns.push(assistant_call(
        "Now wiring main.",
        "call_002",
        "write_file",
        json!({
            "path":"cmd/grpc/main.go",
            "contents":"package main\n\nimport (\n\t\"log\"\n\t\"net\"\n\n\t\"google.golang.org/grpc\"\n\n\torders \"example.com/internal/orders\"\n\tpb \"example.com/proto/orders\"\n)\n\nfunc main() {\n\tlis, err := net.Listen(\"tcp\", \":9000\")\n\tif err != nil {\n\t\tlog.Fatalf(\"listen: %v\", err)\n\t}\n\ts := grpc.NewServer()\n\tpb.RegisterOrdersServer(s, &orders.Server{})\n\tlog.Printf(\"orders gRPC server listening on :9000\")\n\tif err := s.Serve(lis); err != nil {\n\t\tlog.Fatalf(\"serve: %v\", err)\n\t}\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 470 bytes to cmd/grpc/main.go"));
    turns.push(assistant_text(
        "Server is wired and serving. Production deployment should layer TLS via grpc.Creds; left as a follow-up.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_019",
        "go",
        src("github.com/grpc/grpc-go", "Apache-2.0", "d7e3a18"),
        turns,
    )
}

fn go_020_table_test() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Convert the existing tests in str_test.go into a table-driven form using t.Run subtests. Keep behavior identical.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"str_test.go"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package strs\n\nimport \"testing\"\n\nfunc TestReverseEmpty(t *testing.T) {\n\tif got := Reverse(\"\"); got != \"\" { t.Fatalf(\"got %q\", got) }\n}\nfunc TestReverseHello(t *testing.T) {\n\tif got := Reverse(\"hello\"); got != \"olleh\" { t.Fatalf(\"got %q\", got) }\n}\nfunc TestReverseUnicode(t *testing.T) {\n\tif got := Reverse(\"héllo\"); got != \"olléh\" { t.Fatalf(\"got %q\", got) }\n}\n",
    ));
    turns.push(assistant_call(
        "Converting.",
        "call_002",
        "write_file",
        json!({
            "path":"str_test.go",
            "contents":"package strs\n\nimport \"testing\"\n\nfunc TestReverse(t *testing.T) {\n\tcases := []struct {\n\t\tname, in, want string\n\t}{\n\t\t{\"empty\", \"\", \"\"},\n\t\t{\"ascii\", \"hello\", \"olleh\"},\n\t\t{\"unicode\", \"héllo\", \"olléh\"},\n\t}\n\tfor _, tc := range cases {\n\t\ttc := tc\n\t\tt.Run(tc.name, func(t *testing.T) {\n\t\t\tif got := Reverse(tc.in); got != tc.want {\n\t\t\t\tt.Fatalf(\"Reverse(%q) = %q, want %q\", tc.in, got, tc.want)\n\t\t\t}\n\t\t})\n\t}\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 484 bytes to str_test.go"));
    turns.push(assistant_text(
        "Tests are now table-driven; each row gets its own subtest with a descriptive name.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_020",
        "go",
        src("github.com/stretchr/testify", "MIT", "1f2c8e0"),
        turns,
    )
}
