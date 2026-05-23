//! Five long trajectories (11-15 turns each) used to exercise the
//! long-context tail of the AA-AgentPerf distribution.
//!
//! These are not added to `all_tuning_trajectories` directly — each
//! language submodule pulls in the long trajectory it owns so that
//! the per-language counts stay correct (Rust=5, Python=5, ...).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

/// Long Rust trajectory: implement a small in-memory LRU cache from
/// scratch with tests, modeled after the layout of `lru-rs`.
pub fn rust_long() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Implement a fixed-capacity LRU cache in src/lru.rs with insert/get/len, eviction on overflow, and unit tests. Use std::collections::HashMap + a manually maintained doubly-linked list of keys.",
    ));
    turns.push(assistant_call(
        "Listing src/ to see what's there.",
        "call_001",
        "list_dir",
        json!({"path":"src"}),
        &turns,
    ));
    turns.push(tool("call_001", "lib.rs\n"));
    turns.push(assistant_call(
        "Reading lib.rs.",
        "call_002",
        "read_file",
        json!({"path":"src/lib.rs"}),
        &turns,
    ));
    turns.push(tool("call_002", "pub fn placeholder() {}\n"));
    turns.push(assistant_call(
        "Writing the LRU module.",
        "call_003",
        "write_file",
        json!({
            "path":"src/lru.rs",
            "contents":"use std::collections::HashMap;\n\nstruct Node<K, V> { key: K, val: V, prev: Option<usize>, next: Option<usize> }\n\npub struct Lru<K, V> {\n    cap: usize,\n    nodes: Vec<Node<K, V>>,\n    free: Vec<usize>,\n    head: Option<usize>,\n    tail: Option<usize>,\n    map: HashMap<K, usize>,\n}\n\nimpl<K: std::hash::Hash + Eq + Clone, V> Lru<K, V> {\n    pub fn new(cap: usize) -> Self {\n        assert!(cap > 0, \"capacity must be > 0\");\n        Self { cap, nodes: Vec::with_capacity(cap), free: Vec::new(), head: None, tail: None, map: HashMap::with_capacity(cap) }\n    }\n    pub fn len(&self) -> usize { self.map.len() }\n    pub fn is_empty(&self) -> bool { self.len() == 0 }\n    pub fn get(&mut self, k: &K) -> Option<&V> {\n        let idx = *self.map.get(k)?;\n        self.unlink(idx);\n        self.push_front(idx);\n        Some(&self.nodes[idx].val)\n    }\n    pub fn insert(&mut self, k: K, v: V) {\n        if let Some(&idx) = self.map.get(&k) {\n            self.nodes[idx].val = v;\n            self.unlink(idx);\n            self.push_front(idx);\n            return;\n        }\n        if self.len() == self.cap { self.evict(); }\n        let idx = self.alloc(k.clone(), v);\n        self.push_front(idx);\n        self.map.insert(k, idx);\n    }\n    fn alloc(&mut self, key: K, val: V) -> usize {\n        let node = Node { key, val, prev: None, next: None };\n        if let Some(i) = self.free.pop() { self.nodes[i] = node; i }\n        else { self.nodes.push(node); self.nodes.len() - 1 }\n    }\n    fn unlink(&mut self, idx: usize) {\n        let prev = self.nodes[idx].prev;\n        let next = self.nodes[idx].next;\n        if let Some(p) = prev { self.nodes[p].next = next; } else { self.head = next; }\n        if let Some(n) = next { self.nodes[n].prev = prev; } else { self.tail = prev; }\n        self.nodes[idx].prev = None; self.nodes[idx].next = None;\n    }\n    fn push_front(&mut self, idx: usize) {\n        self.nodes[idx].prev = None;\n        self.nodes[idx].next = self.head;\n        if let Some(h) = self.head { self.nodes[h].prev = Some(idx); }\n        self.head = Some(idx);\n        if self.tail.is_none() { self.tail = Some(idx); }\n    }\n    fn evict(&mut self) {\n        let Some(t) = self.tail else { return };\n        self.unlink(t);\n        let key = self.nodes[t].key.clone();\n        self.map.remove(&key);\n        self.free.push(t);\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 2238 bytes to src/lru.rs"));
    turns.push(assistant_call(
        "Adding the module to lib.rs.",
        "call_004",
        "write_file",
        json!({
            "path":"src/lib.rs",
            "contents":"pub mod lru;\npub use lru::Lru;\n"
        }),
        &turns,
    ));
    turns.push(tool("call_004", "wrote 32 bytes to src/lib.rs"));
    turns.push(assistant_call(
        "Now the tests file.",
        "call_005",
        "write_file",
        json!({
            "path":"tests/lru_test.rs",
            "contents":"use demo::Lru;\n\n#[test]\nfn basic_insert_get() {\n    let mut c: Lru<&'static str, i32> = Lru::new(2);\n    c.insert(\"a\", 1);\n    c.insert(\"b\", 2);\n    assert_eq!(c.get(&\"a\"), Some(&1));\n    c.insert(\"c\", 3);\n    assert_eq!(c.get(&\"b\"), None); // b evicted (LRU)\n    assert_eq!(c.get(&\"a\"), Some(&1));\n    assert_eq!(c.get(&\"c\"), Some(&3));\n}\n\n#[test]\nfn update_does_not_grow() {\n    let mut c: Lru<&'static str, i32> = Lru::new(2);\n    c.insert(\"a\", 1);\n    c.insert(\"a\", 2);\n    assert_eq!(c.len(), 1);\n    assert_eq!(c.get(&\"a\"), Some(&2));\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_005", "wrote 530 bytes to tests/lru_test.rs"));
    turns.push(assistant_call(
        "Running the tests.",
        "call_006",
        "run_shell",
        json!({"cmd":"cargo test --tests"}),
        &turns,
    ));
    turns.push(tool(
        "call_006",
        "running 2 tests\ntest basic_insert_get ... ok\ntest update_does_not_grow ... ok\n\ntest result: ok. 2 passed; 0 failed; 0 ignored\n",
    ));
    turns.push(assistant_text(
        "LRU cache shipped with O(1) get/insert and unit tests passing. Eviction strategy is true LRU and the freelist keeps memory usage bounded by capacity.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_051_long",
        "rust",
        src("github.com/jeromefroe/lru-rs", "MIT", "0.12"),
        turns,
    )
}

/// Long Python trajectory: incremental refactor of a small Flask app
/// to use Blueprints + an application factory.
pub fn python_long() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor the single-file Flask app in app.py into an application factory `create_app()` in app/__init__.py with two blueprints: api (in app/api.py) and ui (in app/ui.py). Existing routes go into the matching blueprint.",
    ));
    turns.push(assistant_call(
        "Reading the current app.",
        "call_001",
        "read_file",
        json!({"path":"app.py"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "from flask import Flask, render_template, jsonify\nfrom .db import get_orders\n\napp = Flask(__name__)\n\n@app.route(\"/\")\ndef index(): return render_template(\"index.html\")\n@app.route(\"/dashboard\")\ndef dashboard(): return render_template(\"dashboard.html\")\n@app.route(\"/api/orders\")\ndef api_orders(): return jsonify(get_orders())\n@app.route(\"/api/health\")\ndef api_health(): return {\"status\": \"ok\"}\n",
    ));
    turns.push(assistant_call(
        "Writing the api blueprint.",
        "call_002",
        "write_file",
        json!({
            "path":"app/api.py",
            "contents":"from flask import Blueprint, jsonify\nfrom .db import get_orders\n\napi_bp = Blueprint(\"api\", __name__, url_prefix=\"/api\")\n\n@api_bp.get(\"/orders\")\ndef orders():\n    return jsonify(get_orders())\n\n@api_bp.get(\"/health\")\ndef health():\n    return {\"status\": \"ok\"}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 270 bytes to app/api.py"));
    turns.push(assistant_call(
        "Writing the ui blueprint.",
        "call_003",
        "write_file",
        json!({
            "path":"app/ui.py",
            "contents":"from flask import Blueprint, render_template\n\nui_bp = Blueprint(\"ui\", __name__)\n\n@ui_bp.get(\"/\")\ndef index():\n    return render_template(\"index.html\")\n\n@ui_bp.get(\"/dashboard\")\ndef dashboard():\n    return render_template(\"dashboard.html\")\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 248 bytes to app/ui.py"));
    turns.push(assistant_call(
        "Writing the factory.",
        "call_004",
        "write_file",
        json!({
            "path":"app/__init__.py",
            "contents":"from flask import Flask\n\ndef create_app() -> Flask:\n    app = Flask(__name__)\n    from .api import api_bp\n    from .ui import ui_bp\n    app.register_blueprint(api_bp)\n    app.register_blueprint(ui_bp)\n    return app\n"
        }),
        &turns,
    ));
    turns.push(tool("call_004", "wrote 246 bytes to app/__init__.py"));
    turns.push(assistant_call(
        "Removing the old monolithic file.",
        "call_005",
        "delete_file",
        json!({"path":"app.py"}),
        &turns,
    ));
    turns.push(tool("call_005", "deleted app.py"));
    turns.push(assistant_call(
        "Smoke-testing.",
        "call_006",
        "run_shell",
        json!({"cmd":"FLASK_APP=app:create_app flask routes | head -20"}),
        &turns,
    ));
    turns.push(tool(
        "call_006",
        "Endpoint           Methods  Rule\nui.index           GET      /\nui.dashboard       GET      /dashboard\napi.orders         GET      /api/orders\napi.health         GET      /api/health\nstatic             GET      /static/<path:filename>\n",
    ));
    turns.push(assistant_text(
        "Refactor complete. Routes are unchanged for callers; the factory makes the app testable via `create_app()` and ready for the standard `flask run` invocation.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_052_long",
        "python",
        src("github.com/pallets/flask", "BSD-3-Clause", "d6c5e0b"),
        turns,
    )
}

/// Long TypeScript trajectory: refactor a Next.js page from getServerSideProps
/// to App Router with a streaming server component.
pub fn typescript_long() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Convert pages/projects/[id].tsx from the legacy `getServerSideProps` pattern to an App Router server component at app/projects/[id]/page.tsx, including a `loading.tsx` skeleton.",
    ));
    turns.push(assistant_call(
        "Reading the page.",
        "call_001",
        "read_file",
        json!({"path":"pages/projects/[id].tsx"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import type { GetServerSideProps } from \"next\";\nimport { getProject } from \"../../lib/projects\";\n\nexport default function ProjectPage({ project }: { project: Project }) {\n  return <article><h1>{project.name}</h1><p>{project.description}</p></article>;\n}\n\nexport const getServerSideProps: GetServerSideProps = async (ctx) => {\n  const project = await getProject(ctx.params!.id as string);\n  if (!project) return { notFound: true };\n  return { props: { project } };\n};\n",
    ));
    turns.push(assistant_call(
        "Writing the new page.",
        "call_002",
        "write_file",
        json!({
            "path":"app/projects/[id]/page.tsx",
            "contents":"import { notFound } from \"next/navigation\";\nimport { getProject } from \"@/lib/projects\";\n\ninterface PageProps { params: { id: string } }\n\nexport default async function ProjectPage({ params }: PageProps) {\n  const project = await getProject(params.id);\n  if (!project) notFound();\n  return (\n    <article>\n      <h1>{project.name}</h1>\n      <p>{project.description}</p>\n    </article>\n  );\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 414 bytes to app/projects/[id]/page.tsx"));
    turns.push(assistant_call(
        "Adding the loading skeleton.",
        "call_003",
        "write_file",
        json!({
            "path":"app/projects/[id]/loading.tsx",
            "contents":"export default function Loading() {\n  return (\n    <article aria-busy>\n      <div className=\"h-8 w-1/3 bg-zinc-200 rounded animate-pulse\" />\n      <div className=\"mt-4 h-4 w-full bg-zinc-200 rounded animate-pulse\" />\n      <div className=\"mt-2 h-4 w-2/3 bg-zinc-200 rounded animate-pulse\" />\n    </article>\n  );\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 290 bytes to app/projects/[id]/loading.tsx"));
    turns.push(assistant_call(
        "Deleting the old pages-router file.",
        "call_004",
        "delete_file",
        json!({"path":"pages/projects/[id].tsx"}),
        &turns,
    ));
    turns.push(tool("call_004", "deleted pages/projects/[id].tsx"));
    turns.push(assistant_call(
        "Verifying TypeScript still compiles.",
        "call_005",
        "run_shell",
        json!({"cmd":"npx tsc --noEmit"}),
        &turns,
    ));
    turns.push(tool("call_005", ""));
    turns.push(assistant_call(
        "Running the dev server to confirm the route resolves.",
        "call_006",
        "run_shell",
        json!({"cmd":"curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/projects/42"}),
        &turns,
    ));
    turns.push(tool("call_006", "200"));
    turns.push(assistant_text(
        "Migration complete. The page is now a streaming server component; loading.tsx auto-renders while data resolves; getServerSideProps is gone.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_053_long",
        "typescript",
        src("github.com/vercel/next.js", "MIT", "ab12cd3"),
        turns,
    )
}

/// Long Go trajectory: end-to-end HTTP middleware (logging + recover)
/// with table-driven tests.
pub fn go_long() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add two HTTP middlewares in pkg/middleware: WithRequestID and WithRecover. WithRequestID generates a UUID per request and attaches it to context + response header. WithRecover catches panics and returns 500. Wire both in cmd/server/main.go. Write table-driven tests for each.",
    ));
    turns.push(assistant_call(
        "Looking at the server entry.",
        "call_001",
        "read_file",
        json!({"path":"cmd/server/main.go"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package main\nimport (\"log\"; \"net/http\")\nfunc main() {\n\tmux := http.NewServeMux()\n\tmux.HandleFunc(\"/\", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(\"ok\")) })\n\tlog.Fatal(http.ListenAndServe(\":8080\", mux))\n}\n",
    ));
    turns.push(assistant_call(
        "Writing the middleware package.",
        "call_002",
        "write_file",
        json!({
            "path":"pkg/middleware/middleware.go",
            "contents":"package middleware\n\nimport (\n\t\"context\"\n\t\"log\"\n\t\"net/http\"\n\n\t\"github.com/google/uuid\"\n)\n\ntype ctxKey struct{}\n\nvar requestIDKey ctxKey\n\nfunc WithRequestID(next http.Handler) http.Handler {\n\treturn http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\n\t\tid := r.Header.Get(\"X-Request-Id\")\n\t\tif id == \"\" { id = uuid.NewString() }\n\t\tw.Header().Set(\"X-Request-Id\", id)\n\t\tctx := context.WithValue(r.Context(), requestIDKey, id)\n\t\tnext.ServeHTTP(w, r.WithContext(ctx))\n\t})\n}\n\nfunc RequestID(ctx context.Context) string {\n\tif v, ok := ctx.Value(requestIDKey).(string); ok { return v }\n\treturn \"\"\n}\n\nfunc WithRecover(next http.Handler) http.Handler {\n\treturn http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\n\t\tdefer func() {\n\t\t\tif rec := recover(); rec != nil {\n\t\t\t\tlog.Printf(\"panic in handler: %v\", rec)\n\t\t\t\thttp.Error(w, \"internal server error\", http.StatusInternalServerError)\n\t\t\t}\n\t\t}()\n\t\tnext.ServeHTTP(w, r)\n\t})\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 974 bytes to pkg/middleware/middleware.go"));
    turns.push(assistant_call(
        "Now the tests.",
        "call_003",
        "write_file",
        json!({
            "path":"pkg/middleware/middleware_test.go",
            "contents":"package middleware_test\n\nimport (\n\t\"net/http\"\n\t\"net/http/httptest\"\n\t\"testing\"\n\n\t\"example.com/pkg/middleware\"\n)\n\nfunc TestWithRequestID(t *testing.T) {\n\tcases := []struct {\n\t\tname, header string\n\t\twant         string\n\t}{\n\t\t{\"forwarded\", \"abc-123\", \"abc-123\"},\n\t\t{\"generated\", \"\", \"\"},\n\t}\n\tfor _, tc := range cases {\n\t\ttc := tc\n\t\tt.Run(tc.name, func(t *testing.T) {\n\t\t\th := middleware.WithRequestID(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\n\t\t\t\tif got := middleware.RequestID(r.Context()); tc.want != \"\" && got != tc.want {\n\t\t\t\t\tt.Fatalf(\"got %q want %q\", got, tc.want)\n\t\t\t\t}\n\t\t\t}))\n\t\t\treq := httptest.NewRequest(\"GET\", \"/\", nil)\n\t\t\tif tc.header != \"\" { req.Header.Set(\"X-Request-Id\", tc.header) }\n\t\t\trec := httptest.NewRecorder()\n\t\t\th.ServeHTTP(rec, req)\n\t\t\tif rec.Header().Get(\"X-Request-Id\") == \"\" {\n\t\t\t\tt.Fatalf(\"expected response header to be set\")\n\t\t\t}\n\t\t})\n\t}\n}\n\nfunc TestWithRecover(t *testing.T) {\n\th := middleware.WithRecover(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { panic(\"boom\") }))\n\trec := httptest.NewRecorder()\n\th.ServeHTTP(rec, httptest.NewRequest(\"GET\", \"/\", nil))\n\tif rec.Code != http.StatusInternalServerError {\n\t\tt.Fatalf(\"want 500 got %d\", rec.Code)\n\t}\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 1138 bytes to pkg/middleware/middleware_test.go"));
    turns.push(assistant_call(
        "Wiring the middlewares in main.",
        "call_004",
        "write_file",
        json!({
            "path":"cmd/server/main.go",
            "contents":"package main\n\nimport (\n\t\"log\"\n\t\"net/http\"\n\n\t\"example.com/pkg/middleware\"\n)\n\nfunc main() {\n\tmux := http.NewServeMux()\n\tmux.HandleFunc(\"/\", func(w http.ResponseWriter, r *http.Request) {\n\t\tw.Write([]byte(\"ok request_id=\" + middleware.RequestID(r.Context())))\n\t})\n\tvar h http.Handler = mux\n\th = middleware.WithRecover(h)\n\th = middleware.WithRequestID(h)\n\tlog.Fatal(http.ListenAndServe(\":8080\", h))\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_004", "wrote 444 bytes to cmd/server/main.go"));
    turns.push(assistant_call(
        "Running tests.",
        "call_005",
        "run_shell",
        json!({"cmd":"go test ./pkg/middleware/..."}),
        &turns,
    ));
    turns.push(tool(
        "call_005",
        "ok  \texample.com/pkg/middleware\t0.012s\n",
    ));
    turns.push(assistant_text(
        "Both middlewares wired and tested. Order of composition is important: WithRequestID is outermost so the recover middleware sees a context that already carries the id.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_054_long",
        "go",
        src("github.com/golang/go", "BSD-3-Clause", "5b9c89e"),
        turns,
    )
}

/// Long Java trajectory: Spring Boot service for a paginated endpoint
/// + DTO + unit test.
pub fn java_long() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a GET /api/orders endpoint to OrdersController that returns Page<OrderDTO> using Spring Data's Pageable. Map from Order entity via OrderMapper. Add a slice test for the controller.",
    ));
    turns.push(assistant_call(
        "Reading the controller.",
        "call_001",
        "read_file",
        json!({"path":"src/main/java/com/acme/orders/OrdersController.java"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package com.acme.orders;\nimport org.springframework.web.bind.annotation.*;\n\n@RestController\n@RequestMapping(\"/api/orders\")\npublic class OrdersController {\n    private final OrdersService service;\n    public OrdersController(OrdersService service) { this.service = service; }\n}\n",
    ));
    turns.push(assistant_call(
        "Looking at the service + repo.",
        "call_002",
        "read_file",
        json!({"path":"src/main/java/com/acme/orders/OrdersService.java"}),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "package com.acme.orders;\nimport org.springframework.data.domain.*;\nimport org.springframework.stereotype.Service;\n\n@Service\npublic class OrdersService {\n    private final OrderRepository repo;\n    public OrdersService(OrderRepository repo) { this.repo = repo; }\n    public Page<Order> findAll(Pageable p) { return repo.findAll(p); }\n}\n",
    ));
    turns.push(assistant_call(
        "Reading the mapper.",
        "call_003",
        "read_file",
        json!({"path":"src/main/java/com/acme/orders/OrderMapper.java"}),
        &turns,
    ));
    turns.push(tool(
        "call_003",
        "package com.acme.orders;\n\npublic final class OrderMapper {\n    private OrderMapper() {}\n    public static OrderDTO toDTO(Order o) { return new OrderDTO(o.getId(), o.getTotalCents(), o.getCreatedAt()); }\n}\n",
    ));
    turns.push(assistant_call(
        "Writing the controller method.",
        "call_004",
        "write_file",
        json!({
            "path":"src/main/java/com/acme/orders/OrdersController.java",
            "contents":"package com.acme.orders;\nimport org.springframework.data.domain.Page;\nimport org.springframework.data.domain.Pageable;\nimport org.springframework.web.bind.annotation.*;\n\n@RestController\n@RequestMapping(\"/api/orders\")\npublic class OrdersController {\n    private final OrdersService service;\n    public OrdersController(OrdersService service) { this.service = service; }\n\n    @GetMapping\n    public Page<OrderDTO> list(Pageable pageable) {\n        return service.findAll(pageable).map(OrderMapper::toDTO);\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_004",
        "wrote 540 bytes to src/main/java/com/acme/orders/OrdersController.java",
    ));
    turns.push(assistant_call(
        "Adding the controller slice test.",
        "call_005",
        "write_file",
        json!({
            "path":"src/test/java/com/acme/orders/OrdersControllerTest.java",
            "contents":"package com.acme.orders;\nimport org.junit.jupiter.api.Test;\nimport org.mockito.Mockito;\nimport org.springframework.beans.factory.annotation.Autowired;\nimport org.springframework.boot.test.context.SpringBootTest;\nimport org.springframework.boot.test.mock.mockito.MockBean;\nimport org.springframework.data.domain.*;\nimport org.springframework.test.web.servlet.MockMvc;\nimport org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;\n\nimport java.time.Instant;\nimport java.util.List;\n\nimport static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;\nimport static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;\n\n@SpringBootTest\n@AutoConfigureMockMvc\nclass OrdersControllerTest {\n    @Autowired MockMvc mvc;\n    @MockBean OrdersService service;\n\n    @Test\n    void getOrders_returnsPage() throws Exception {\n        Order o = new Order(1L, 1099L, Instant.now());\n        Mockito.when(service.findAll(Mockito.any()))\n            .thenReturn(new PageImpl<>(List.of(o), PageRequest.of(0, 20), 1));\n        mvc.perform(get(\"/api/orders\"))\n            .andExpect(status().isOk())\n            .andExpect(jsonPath(\"$.content[0].id\").value(1));\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_005",
        "wrote 1226 bytes to src/test/java/com/acme/orders/OrdersControllerTest.java",
    ));
    turns.push(assistant_call(
        "Running the test.",
        "call_006",
        "run_shell",
        json!({"cmd":"./mvnw test -Dtest=OrdersControllerTest -q"}),
        &turns,
    ));
    turns.push(tool(
        "call_006",
        "[INFO] BUILD SUCCESS\n",
    ));
    turns.push(assistant_text(
        "Paginated endpoint shipped with a controller test. Add a json:date deserializer if the Instant payload field name in OrderDTO needs a custom format.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_055_long",
        "java",
        src("github.com/spring-projects/spring-boot", "Apache-2.0", "9c7a234"),
        turns,
    )
}
