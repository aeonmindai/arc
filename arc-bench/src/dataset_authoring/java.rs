//! Java trajectories (4).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        java_027_optional_chain(),
        java_028_stream_collect(),
        java_029_junit_parameterized(),
        super::long_trajectories::java_long(),
    ]
}

#[allow(dead_code)]
fn java_026_spring_controller() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a Spring Boot REST controller HealthController at src/main/java/com/acme/api/HealthController.java exposing GET /healthz returning {\"status\":\"ok\"}.",
    ));
    turns.push(assistant_call(
        "Reading the package layout.",
        "call_001",
        "list_dir",
        json!({"path":"src/main/java/com/acme/api"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "UsersController.java\nProjectsController.java\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_002",
        "write_file",
        json!({
            "path":"src/main/java/com/acme/api/HealthController.java",
            "contents":"package com.acme.api;\n\nimport java.util.Map;\nimport org.springframework.web.bind.annotation.GetMapping;\nimport org.springframework.web.bind.annotation.RestController;\n\n@RestController\npublic class HealthController {\n    @GetMapping(\"/healthz\")\n    public Map<String, String> healthz() {\n        return Map.of(\"status\", \"ok\");\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 358 bytes to src/main/java/com/acme/api/HealthController.java",
    ));
    turns.push(assistant_text(
        "Controller registered via component scan; GET /healthz returns the JSON map.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_026",
        "java",
        src("github.com/spring-projects/spring-boot", "Apache-2.0", "9c7a234"),
        turns,
    )
}

fn java_027_optional_chain() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "In UserService.findEmail(long id), refactor the nested if/else null checks into a single Optional chain.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/main/java/com/acme/users/UserService.java"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package com.acme.users;\nimport java.util.Optional;\n\npublic class UserService {\n    private final UserRepo repo;\n    public UserService(UserRepo r) { this.repo = r; }\n\n    public String findEmail(long id) {\n        User u = repo.find(id);\n        if (u != null) {\n            Profile p = u.profile();\n            if (p != null) {\n                String e = p.email();\n                if (e != null && !e.isBlank()) return e.toLowerCase();\n            }\n        }\n        return \"unknown\";\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "Refactoring.",
        "call_002",
        "write_file",
        json!({
            "path":"src/main/java/com/acme/users/UserService.java",
            "contents":"package com.acme.users;\nimport java.util.Optional;\n\npublic class UserService {\n    private final UserRepo repo;\n    public UserService(UserRepo r) { this.repo = r; }\n\n    public String findEmail(long id) {\n        return Optional.ofNullable(repo.find(id))\n                .map(User::profile)\n                .map(Profile::email)\n                .filter(e -> !e.isBlank())\n                .map(String::toLowerCase)\n                .orElse(\"unknown\");\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 414 bytes to src/main/java/com/acme/users/UserService.java",
    ));
    turns.push(assistant_text(
        "Single Optional chain; behavior identical.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_027",
        "java",
        src("github.com/google/guava", "Apache-2.0", "4d2f9c1"),
        turns,
    )
}

fn java_028_stream_collect() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor groupOrdersByCustomer in OrderReport.java from a manual HashMap loop into a Stream.collect(Collectors.groupingBy).",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/main/java/com/acme/reports/OrderReport.java"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package com.acme.reports;\nimport java.util.*;\n\npublic final class OrderReport {\n    public static Map<Long, List<Order>> groupOrdersByCustomer(List<Order> orders) {\n        Map<Long, List<Order>> out = new HashMap<>();\n        for (Order o : orders) {\n            out.computeIfAbsent(o.customerId(), k -> new ArrayList<>()).add(o);\n        }\n        return out;\n    }\n}\n",
    ));
    turns.push(assistant_call(
        "Refactoring.",
        "call_002",
        "write_file",
        json!({
            "path":"src/main/java/com/acme/reports/OrderReport.java",
            "contents":"package com.acme.reports;\nimport java.util.*;\nimport java.util.stream.Collectors;\n\npublic final class OrderReport {\n    public static Map<Long, List<Order>> groupOrdersByCustomer(List<Order> orders) {\n        return orders.stream().collect(Collectors.groupingBy(Order::customerId));\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 332 bytes to src/main/java/com/acme/reports/OrderReport.java",
    ));
    turns.push(assistant_text(
        "Grouping is now a single stream collector; behavior preserved.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_028",
        "java",
        src("github.com/openjdk/jdk", "GPL-2.0-with-classpath-exception", "21+35"),
        turns,
    )
}

fn java_029_junit_parameterized() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Convert StringsTest.reverse_examples (three @Test methods) into a single @ParameterizedTest with @CsvSource.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/test/java/com/acme/strings/StringsTest.java"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package com.acme.strings;\nimport org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.assertEquals;\n\npublic class StringsTest {\n    @Test void reverse_empty() { assertEquals(\"\", Strings.reverse(\"\")); }\n    @Test void reverse_ascii() { assertEquals(\"olleh\", Strings.reverse(\"hello\")); }\n    @Test void reverse_unicode() { assertEquals(\"olléh\", Strings.reverse(\"héllo\")); }\n}\n",
    ));
    turns.push(assistant_call(
        "Writing the parameterized version.",
        "call_002",
        "write_file",
        json!({
            "path":"src/test/java/com/acme/strings/StringsTest.java",
            "contents":"package com.acme.strings;\nimport org.junit.jupiter.params.ParameterizedTest;\nimport org.junit.jupiter.params.provider.CsvSource;\nimport static org.junit.jupiter.api.Assertions.assertEquals;\n\npublic class StringsTest {\n    @ParameterizedTest(name = \"reverse({0}) = {1}\")\n    @CsvSource({\n        \"'', ''\",\n        \"hello, olleh\",\n        \"'héllo', 'olléh'\"\n    })\n    void reverse_examples(String in, String want) {\n        assertEquals(want, Strings.reverse(in));\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 528 bytes to src/test/java/com/acme/strings/StringsTest.java",
    ));
    turns.push(assistant_text(
        "Three test methods collapsed into one parameterized test with three CSV rows.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_029",
        "java",
        src("github.com/junit-team/junit5", "EPL-2.0", "5e8f3c4"),
        turns,
    )
}
