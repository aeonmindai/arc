//! C# trajectories (3).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        cs_048_minimal_api(),
        cs_049_ef_migration(),
        cs_050_xunit_theory(),
    ]
}

fn cs_048_minimal_api() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a Minimal API endpoint `GET /healthz` in Program.cs that returns Results.Ok(new { status = \"ok\" }).",
    ));
    turns.push(assistant_call(
        "Reading Program.cs.",
        "call_001",
        "read_file",
        json!({"path":"Program.cs"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "var builder = WebApplication.CreateBuilder(args);\nvar app = builder.Build();\napp.MapGet(\"/\", () => \"Hello\");\napp.Run();\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_002",
        "write_file",
        json!({
            "path":"Program.cs",
            "contents":"var builder = WebApplication.CreateBuilder(args);\nvar app = builder.Build();\napp.MapGet(\"/\", () => \"Hello\");\napp.MapGet(\"/healthz\", () => Results.Ok(new { status = \"ok\" }));\napp.Run();\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 220 bytes to Program.cs"));
    turns.push(assistant_text("Endpoint registered.", &turns));

    assemble(
        "agentperf_tuning_048",
        "csharp",
        src("github.com/dotnet/aspnetcore", "MIT", "8.0"),
        turns,
    )
}

fn cs_049_ef_migration() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a `LastLoginAt` DateTime? column to the User entity and produce an EF Core migration named AddUserLastLoginAt.",
    ));
    turns.push(assistant_call(
        "Reading the entity.",
        "call_001",
        "read_file",
        json!({"path":"Models/User.cs"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "namespace Acme.Models;\npublic class User {\n    public long Id { get; set; }\n    public string Email { get; set; } = \"\";\n}\n",
    ));
    turns.push(assistant_call(
        "Updating the entity.",
        "call_002",
        "write_file",
        json!({
            "path":"Models/User.cs",
            "contents":"namespace Acme.Models;\npublic class User {\n    public long Id { get; set; }\n    public string Email { get; set; } = \"\";\n    public DateTime? LastLoginAt { get; set; }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 198 bytes to Models/User.cs"));
    turns.push(assistant_call(
        "Generating the migration.",
        "call_003",
        "run_shell",
        json!({"cmd":"dotnet ef migrations add AddUserLastLoginAt"}),
        &turns,
    ));
    turns.push(tool(
        "call_003",
        "Build succeeded.\nDone. To undo this action, use 'ef migrations remove'\n",
    ));
    turns.push(assistant_text(
        "Migration generated under Migrations/. Apply with `dotnet ef database update`.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_049",
        "csharp",
        src("github.com/dotnet/efcore", "MIT", "8.0"),
        turns,
    )
}

fn cs_050_xunit_theory() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Convert StringReverseTests.cs from three Fact methods into a single xUnit Theory backed by InlineData.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"tests/StringReverseTests.cs"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "using Xunit;\npublic class StringReverseTests {\n  [Fact] public void Empty()  { Assert.Equal(\"\", StringUtil.Reverse(\"\")); }\n  [Fact] public void Ascii()  { Assert.Equal(\"olleh\", StringUtil.Reverse(\"hello\")); }\n  [Fact] public void Accent() { Assert.Equal(\"olléh\", StringUtil.Reverse(\"héllo\")); }\n}\n",
    ));
    turns.push(assistant_call(
        "Refactoring.",
        "call_002",
        "write_file",
        json!({
            "path":"tests/StringReverseTests.cs",
            "contents":"using Xunit;\npublic class StringReverseTests {\n  [Theory]\n  [InlineData(\"\", \"\")]\n  [InlineData(\"hello\", \"olleh\")]\n  [InlineData(\"héllo\", \"olléh\")]\n  public void Reverse_examples(string input, string expected) {\n    Assert.Equal(expected, StringUtil.Reverse(input));\n  }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 308 bytes to tests/StringReverseTests.cs",
    ));
    turns.push(assistant_text(
        "Three facts collapsed into one Theory with three rows.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_050",
        "csharp",
        src("github.com/xunit/xunit", "Apache-2.0", "2.9"),
        turns,
    )
}
