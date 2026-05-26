//! Scala trajectories (3).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        sc_045_cats_effect_resource(),
        sc_046_play_action(),
        sc_047_zio_layer(),
    ]
}

fn sc_045_cats_effect_resource() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor HttpServer.start in App.scala to acquire/release its server binding through cats.effect.Resource so it shuts down on cancel.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/main/scala/App.scala"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import cats.effect.{IO, IOApp}\n\nobject App extends IOApp.Simple {\n  def run: IO[Unit] = HttpServer.start.flatMap(_ => IO.never)\n}\n",
    ));
    turns.push(assistant_call(
        "Reading the server.",
        "call_002",
        "read_file",
        json!({"path":"src/main/scala/HttpServer.scala"}),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "import cats.effect.IO\nimport org.http4s.ember.server.EmberServerBuilder\nimport com.comcast.ip4s.*\n\nobject HttpServer {\n  def start: IO[Unit] =\n    EmberServerBuilder.default[IO].withHost(host\"0.0.0.0\").withPort(port\"8080\").build.use(_ => IO.never)\n}\n",
    ));
    turns.push(assistant_call(
        "Switching to Resource.",
        "call_003",
        "write_file",
        json!({
            "path":"src/main/scala/HttpServer.scala",
            "contents":"import cats.effect.{IO, Resource}\nimport com.comcast.ip4s.*\nimport org.http4s.server.Server\nimport org.http4s.ember.server.EmberServerBuilder\n\nobject HttpServer {\n  val resource: Resource[IO, Server] =\n    EmberServerBuilder.default[IO]\n      .withHost(host\"0.0.0.0\")\n      .withPort(port\"8080\")\n      .build\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_003",
        "wrote 348 bytes to src/main/scala/HttpServer.scala",
    ));
    turns.push(assistant_call(
        "Updating App.scala to consume the Resource.",
        "call_004",
        "write_file",
        json!({
            "path":"src/main/scala/App.scala",
            "contents":"import cats.effect.{IO, IOApp}\n\nobject App extends IOApp.Simple {\n  def run: IO[Unit] = HttpServer.resource.use(_ => IO.never)\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_004",
        "wrote 156 bytes to src/main/scala/App.scala",
    ));
    turns.push(assistant_text(
        "Server is now Resource-managed; cancelling the IO releases the binding via Ember's shutdown.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_045",
        "scala",
        src("github.com/typelevel/cats-effect", "Apache-2.0", "3.5.x"),
        turns,
    )
}

fn sc_046_play_action() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a JSON action to HomeController for POST /api/users that validates a JsValue against UserForm.reads and returns 201 with the created user, or 400 on invalid input.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"app/controllers/HomeController.scala"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "package controllers\nimport javax.inject._\nimport play.api.mvc._\n\n@Singleton\nclass HomeController @Inject() (cc: ControllerComponents) extends AbstractController(cc) {\n  def index() = Action { Ok(\"hello\") }\n}\n",
    ));
    turns.push(assistant_call(
        "Reading the form.",
        "call_002",
        "read_file",
        json!({"path":"app/forms/UserForm.scala"}),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "package forms\nimport play.api.libs.json._\nfinal case class UserForm(name: String, email: String)\nobject UserForm { implicit val reads: Reads[UserForm] = Json.reads[UserForm]; implicit val writes: Writes[UserForm] = Json.writes[UserForm] }\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_003",
        "write_file",
        json!({
            "path":"app/controllers/HomeController.scala",
            "contents":"package controllers\nimport javax.inject._\nimport play.api.libs.json._\nimport play.api.mvc._\nimport forms.UserForm\n\n@Singleton\nclass HomeController @Inject() (cc: ControllerComponents) extends AbstractController(cc) {\n  def index() = Action { Ok(\"hello\") }\n\n  def createUser() = Action(parse.json) { request =>\n    request.body.validate[UserForm].fold(\n      errs => BadRequest(JsError.toJson(errs)),\n      form => Created(Json.toJson(form))\n    )\n  }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_003",
        "wrote 564 bytes to app/controllers/HomeController.scala",
    ));
    turns.push(assistant_text(
        "Route Play config needs `POST /api/users controllers.HomeController.createUser` added to conf/routes.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_046",
        "scala",
        src(
            "github.com/playframework/playframework",
            "Apache-2.0",
            "3.0",
        ),
        turns,
    )
}

fn sc_047_zio_layer() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Wire a ZLayer for OrdersService backed by a Postgres ConnectionPool. Provide the live layer in Main.run.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"src/main/scala/OrdersService.scala"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "import zio._\n\ntrait OrdersService { def list(): UIO[List[Order]] }\nfinal class LiveOrdersService(pool: ConnectionPool) extends OrdersService {\n  def list(): UIO[List[Order]] = ZIO.succeed(Nil)\n}\nobject OrdersService {\n  val live: ZLayer[ConnectionPool, Nothing, OrdersService] =\n    ZLayer.fromFunction(new LiveOrdersService(_))\n}\n",
    ));
    turns.push(assistant_call(
        "Reading Main.",
        "call_002",
        "read_file",
        json!({"path":"src/main/scala/Main.scala"}),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "import zio._\n\nobject Main extends ZIOAppDefault {\n  def run = ZIO.unit\n}\n",
    ));
    turns.push(assistant_call(
        "Wiring layers.",
        "call_003",
        "write_file",
        json!({
            "path":"src/main/scala/Main.scala",
            "contents":"import zio._\n\nobject Main extends ZIOAppDefault {\n  val program = for {\n    svc    <- ZIO.service[OrdersService]\n    orders <- svc.list()\n    _      <- Console.printLine(s\"orders=${orders.size}\")\n  } yield ()\n\n  def run = program.provide(\n    ConnectionPool.live,\n    OrdersService.live\n  )\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_003",
        "wrote 366 bytes to src/main/scala/Main.scala",
    ));
    turns.push(assistant_text(
        "Layers composed; Main now resolves OrdersService through ConnectionPool.live.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_047",
        "scala",
        src("github.com/zio/zio", "Apache-2.0", "2.x"),
        turns,
    )
}
