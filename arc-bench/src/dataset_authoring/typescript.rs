//! TypeScript trajectories (5).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        ts_011_react_usequery(),
        ts_013_zod_schema(),
        ts_014_express_middleware(),
        ts_015_prisma_transaction(),
        super::long_trajectories::typescript_long(),
    ]
}

fn ts_011_react_usequery() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "In src/hooks/useProjects.ts switch the manual fetch to @tanstack/react-query. Return { projects, isLoading, error } and gate refetch behind a 30s staleTime.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/hooks/useProjects.ts"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import { useState, useEffect } from \"react\";\nimport type { Project } from \"../types\";\n\nexport function useProjects() {\n  const [projects, setProjects] = useState<Project[]>([]);\n  const [isLoading, setLoading] = useState(true);\n  const [error, setError] = useState<unknown>(null);\n\n  useEffect(() => {\n    fetch(\"/api/projects\")\n      .then((r) => r.json())\n      .then(setProjects)\n      .catch(setError)\n      .finally(() => setLoading(false));\n  }, []);\n\n  return { projects, isLoading, error };\n}\n",
    ));
    turns.push(assistant_call(
        "Switching to useQuery.",
        "call_002",
        "write_file",
        json!({
            "path":"src/hooks/useProjects.ts",
            "contents":"import { useQuery } from \"@tanstack/react-query\";\nimport type { Project } from \"../types\";\n\nasync function fetchProjects(): Promise<Project[]> {\n  const r = await fetch(\"/api/projects\");\n  if (!r.ok) {\n    throw new Error(`projects fetch failed: ${r.status}`);\n  }\n  return (await r.json()) as Project[];\n}\n\nexport function useProjects() {\n  const { data, isLoading, error } = useQuery({\n    queryKey: [\"projects\"],\n    queryFn: fetchProjects,\n    staleTime: 30_000,\n  });\n  return { projects: data ?? [], isLoading, error };\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 552 bytes to src/hooks/useProjects.ts"));
    turns.push(assistant_text(
        "useProjects now backed by react-query with a 30s staleTime; the return shape is preserved.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_011",
        "typescript",
        src(
            "github.com/TanStack/query",
            "MIT",
            "6d4a8c1",
        ),
        turns,
    )
}

#[allow(dead_code)]
fn ts_012_nextjs_route_handler() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a Next.js App Router route handler at app/api/healthz/route.ts that returns 200 with {ok: true, ts: now} as JSON. Force dynamic.",
    ));
    turns.push(assistant_call(
        "Listing app/api to confirm the layout.",
        "call_001",
        "list_dir",
        json!({"path":"app/api"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "users/\n  route.ts\nprojects/\n  route.ts\n",
    ));
    turns.push(assistant_call(
        "Writing the route.",
        "call_002",
        "write_file",
        json!({
            "path":"app/api/healthz/route.ts",
            "contents":"import { NextResponse } from \"next/server\";\n\nexport const dynamic = \"force-dynamic\";\n\nexport async function GET() {\n  return NextResponse.json({ ok: true, ts: Date.now() });\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 180 bytes to app/api/healthz/route.ts"));
    turns.push(assistant_text(
        "GET /api/healthz now returns {ok, ts} dynamically (no cache).",
        &turns,
    ));

    assemble(
        "agentperf_tuning_012",
        "typescript",
        src("github.com/vercel/next.js", "MIT", "ab12cd3"),
        turns,
    )
}

fn ts_013_zod_schema() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a Zod schema for CreateUser in schemas/user.ts. Email must be valid, age must be int>=18, role enum admin|editor|viewer. Export the inferred TS type.",
    ));
    turns.push(assistant_call(
        "Writing the schema.",
        "call_001",
        "write_file",
        json!({
            "path":"schemas/user.ts",
            "contents":"import { z } from \"zod\";\n\nexport const CreateUserSchema = z.object({\n  email: z.string().email(),\n  age: z.number().int().gte(18),\n  role: z.enum([\"admin\", \"editor\", \"viewer\"]),\n});\n\nexport type CreateUser = z.infer<typeof CreateUserSchema>;\n"
        }),
        &turns,
    ));
    turns.push(tool("call_001", "wrote 268 bytes to schemas/user.ts"));
    turns.push(assistant_text(
        "Schema and inferred type exported.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_013",
        "typescript",
        src("github.com/colinhacks/zod", "MIT", "3f76a4e"),
        turns,
    )
}

fn ts_014_express_middleware() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add an Express request-id middleware in middlewares/requestId.ts that pulls X-Request-Id from the header (or generates a UUID) and attaches it to req.id. Wire it before all routes in server.ts.",
    ));
    turns.push(assistant_call(
        "Reading server.ts to see the current middleware order.",
        "call_001",
        "read_file",
        json!({"path":"server.ts"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import express from \"express\";\nimport { router } from \"./router\";\n\nexport const app = express();\napp.use(express.json());\napp.use(router);\n",
    ));
    turns.push(assistant_call(
        "Writing the middleware.",
        "call_002",
        "write_file",
        json!({
            "path":"middlewares/requestId.ts",
            "contents":"import type { Request, Response, NextFunction } from \"express\";\nimport { randomUUID } from \"node:crypto\";\n\ndeclare module \"express-serve-static-core\" {\n  interface Request {\n    id?: string;\n  }\n}\n\nexport function requestId(req: Request, res: Response, next: NextFunction) {\n  const header = req.header(\"x-request-id\");\n  const id = header && header.length > 0 ? header : randomUUID();\n  req.id = id;\n  res.setHeader(\"x-request-id\", id);\n  next();\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 514 bytes to middlewares/requestId.ts"));
    turns.push(assistant_call(
        "Wiring it into server.ts.",
        "call_003",
        "write_file",
        json!({
            "path":"server.ts",
            "contents":"import express from \"express\";\nimport { router } from \"./router\";\nimport { requestId } from \"./middlewares/requestId\";\n\nexport const app = express();\napp.use(requestId);\napp.use(express.json());\napp.use(router);\n"
        }),
        &turns,
    ));
    turns.push(tool("call_003", "wrote 250 bytes to server.ts"));
    turns.push(assistant_text(
        "Middleware in place; every request now carries an id on req.id and is echoed in the response header.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_014",
        "typescript",
        src("github.com/expressjs/express", "MIT", "7c8b9a0"),
        turns,
    )
}

fn ts_015_prisma_transaction() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "In services/billing.ts, wrap chargeOrder and writeLedger in a Prisma interactive transaction with isolation level Serializable. Roll back on either failure.",
    ));
    turns.push(assistant_call(
        "Reading the service.",
        "call_001",
        "read_file",
        json!({"path":"services/billing.ts"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import { prisma } from \"../db\";\n\nexport async function chargeAndLedger(orderId: string, amountCents: number) {\n  await prisma.charge.create({ data: { orderId, amountCents } });\n  await prisma.ledger.create({ data: { orderId, amountCents, kind: \"charge\" } });\n}\n",
    ));
    turns.push(assistant_call(
        "Wrapping in $transaction.",
        "call_002",
        "write_file",
        json!({
            "path":"services/billing.ts",
            "contents":"import { Prisma } from \"@prisma/client\";\nimport { prisma } from \"../db\";\n\nexport async function chargeAndLedger(orderId: string, amountCents: number) {\n  return prisma.$transaction(\n    async (tx) => {\n      await tx.charge.create({ data: { orderId, amountCents } });\n      await tx.ledger.create({ data: { orderId, amountCents, kind: \"charge\" } });\n    },\n    { isolationLevel: Prisma.TransactionIsolationLevel.Serializable }\n  );\n}\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 458 bytes to services/billing.ts"));
    turns.push(assistant_text(
        "chargeAndLedger now runs in a single serializable interactive transaction; either insert failing rolls back the other.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_015",
        "typescript",
        src("github.com/prisma/prisma", "Apache-2.0", "f4e7c8a"),
        turns,
    )
}
