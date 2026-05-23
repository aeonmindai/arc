//! Kotlin trajectories (3).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        kt_042_coroutine_flow(),
        kt_043_ktor_route(),
        kt_044_data_class(),
    ]
}

fn kt_042_coroutine_flow() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor TickerService.observe(): Job into a kotlinx Flow<Int> that emits a count every second using `flow { while (true) { emit(...); delay(1000) } }`.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/main/kotlin/com/acme/ticker/TickerService.kt"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package com.acme.ticker\nimport kotlinx.coroutines.*\n\nclass TickerService(private val scope: CoroutineScope) {\n    fun observe(onTick: (Int) -> Unit): Job = scope.launch {\n        var i = 0\n        while (isActive) { onTick(i++); delay(1000) }\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_002",
        "write_file",
        json!({
            "path":"src/main/kotlin/com/acme/ticker/TickerService.kt",
            "contents":"package com.acme.ticker\nimport kotlinx.coroutines.delay\nimport kotlinx.coroutines.flow.Flow\nimport kotlinx.coroutines.flow.flow\n\nclass TickerService {\n    fun observe(): Flow<Int> = flow {\n        var i = 0\n        while (true) {\n            emit(i)\n            i += 1\n            delay(1000)\n        }\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 386 bytes to src/main/kotlin/com/acme/ticker/TickerService.kt"));
    turns.push(assistant_text(
        "Service now exposes a cold Flow; callers control lifecycle via collect/cancel.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_042",
        "kotlin",
        src("github.com/Kotlin/kotlinx.coroutines", "Apache-2.0", "1.8.x"),
        turns,
    )
}

fn kt_043_ktor_route() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a Ktor route GET /healthz under module() that responds with HttpStatusCode.OK and a JSON {status: \"ok\"} using ContentNegotiation/JSON.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/main/kotlin/com/acme/Application.kt"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package com.acme\nimport io.ktor.server.application.*\nimport io.ktor.server.engine.*\nimport io.ktor.server.netty.*\nimport io.ktor.server.routing.*\n\nfun main() { embeddedServer(Netty, port = 8080) { module() }.start(wait = true) }\nfun Application.module() { routing { /* routes */ } }\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_002",
        "write_file",
        json!({
            "path":"src/main/kotlin/com/acme/Application.kt",
            "contents":"package com.acme\nimport io.ktor.http.*\nimport io.ktor.serialization.kotlinx.json.*\nimport io.ktor.server.application.*\nimport io.ktor.server.engine.*\nimport io.ktor.server.netty.*\nimport io.ktor.server.plugins.contentnegotiation.*\nimport io.ktor.server.response.*\nimport io.ktor.server.routing.*\nimport kotlinx.serialization.Serializable\n\n@Serializable data class Health(val status: String)\n\nfun main() { embeddedServer(Netty, port = 8080) { module() }.start(wait = true) }\n\nfun Application.module() {\n    install(ContentNegotiation) { json() }\n    routing {\n        get(\"/healthz\") { call.respond(HttpStatusCode.OK, Health(\"ok\")) }\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 762 bytes to src/main/kotlin/com/acme/Application.kt"));
    turns.push(assistant_text(
        "Healthz wired through Ktor's ContentNegotiation/JSON plugin.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_043",
        "kotlin",
        src("github.com/ktorio/ktor", "Apache-2.0", "2.3"),
        turns,
    )
}

fn kt_044_data_class() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor the User class in domain/User.kt from an explicit-equals/hashCode/toString form to an idiomatic Kotlin data class.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/main/kotlin/com/acme/domain/User.kt"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package com.acme.domain\n\nclass User(val id: Long, val name: String, val email: String) {\n    override fun equals(other: Any?): Boolean { /* manual */ return other is User && other.id == id }\n    override fun hashCode(): Int = id.hashCode()\n    override fun toString(): String = \"User($id, $name, $email)\"\n}\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_002",
        "write_file",
        json!({
            "path":"src/main/kotlin/com/acme/domain/User.kt",
            "contents":"package com.acme.domain\n\ndata class User(val id: Long, val name: String, val email: String)\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 80 bytes to src/main/kotlin/com/acme/domain/User.kt"));
    turns.push(assistant_text(
        "Note that the new equals/hashCode is structural (covers all three fields), unlike the previous id-only equality. Confirm that's what callers expect.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_044",
        "kotlin",
        src("github.com/JetBrains/kotlin", "Apache-2.0", "1.9.20"),
        turns,
    )
}
